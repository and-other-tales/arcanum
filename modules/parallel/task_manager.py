#!/usr/bin/env python3
"""
Parallel Task Manager for Arcanum
--------------------------------
This module provides a parallel task management system for computationally intensive
operations in Arcanum, allowing for better resource utilization and faster processing.
"""

import os
import sys
import logging
import threading
import queue
import time
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
import multiprocessing

# Configure logging
logger = logging.getLogger(__name__)

class Task:
    """Represents a single task in the task management system."""
    
    def __init__(self, task_id: str, name: str, func: Callable, args: Tuple = None, 
                 kwargs: Dict = None, priority: int = 0, retries: int = 0, 
                 timeout: int = None, dependencies: List[str] = None):
        """
        Initialize a new task.
        
        Args:
            task_id: Unique identifier for the task
            name: Human-readable name for the task
            func: The function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Task priority (higher values = higher priority)
            retries: Number of retry attempts if the task fails
            timeout: Timeout in seconds
            dependencies: List of task IDs that must complete before this task
        """
        self.task_id = task_id
        self.name = name
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.priority = priority
        self.retries = retries
        self.timeout = timeout
        self.dependencies = dependencies or []
        
        # State tracking
        self.status = "pending"  # pending, running, completed, failed
        self.start_time = None
        self.end_time = None
        self.attempts = 0
        self.result = None
        self.error = None
        self.progress = 0.0  # 0.0 to 1.0
        self.logs = []
        self.future = None  # For tracking the future when using executors
        
    def to_dict(self) -> Dict:
        """Convert task to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status,
            "priority": self.priority,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "attempts": self.attempts,
            "retries": self.retries,
            "progress": self.progress,
            "error": str(self.error) if self.error else None,
            "dependencies": self.dependencies,
            "has_result": self.result is not None
        }
        
    def __lt__(self, other):
        """Comparison for priority queue."""
        if not isinstance(other, Task):
            return NotImplemented
        return self.priority > other.priority  # Higher priority comes first
        
        
class TaskProgressCallback:
    """Callback object for reporting task progress."""
    
    def __init__(self, task_id: str, task_manager = None):
        """
        Initialize a progress callback for a task.
        
        Args:
            task_id: ID of the task to report progress for
            task_manager: Optional task manager reference for direct updates
        """
        self.task_id = task_id
        self.task_manager = task_manager
        
    def update(self, progress: float, message: str = None):
        """
        Update progress for the task.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            message: Optional message about the progress
        """
        if self.task_manager:
            self.task_manager.update_task_progress(self.task_id, progress, message)
        else:
            logger.info(f"Task {self.task_id} progress: {progress:.1%} {message or ''}")


class TaskManager:
    """Manages the execution of tasks with dependencies and parallel processing."""
    
    def __init__(self, max_workers: int = None, use_processes: bool = False, 
                 max_retries: int = 3, default_timeout: int = 3600):
        """
        Initialize the task manager.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: If True, use ProcessPoolExecutor instead of ThreadPoolExecutor
            max_retries: Default maximum retry attempts for failed tasks
            default_timeout: Default timeout in seconds
        """
        # Determine number of workers based on CPU cores
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 4) + 4)
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.max_retries = max_retries
        self.default_timeout = default_timeout
        
        # Task storage and tracking
        self.tasks = {}  # task_id -> Task
        self.task_queue = queue.PriorityQueue()
        self.running_tasks = set()
        self.completed_tasks = set()
        self.failed_tasks = set()
        
        # Executor - we'll initialize this only when needed
        self.executor = None
        
        # Threading control
        self.running = False
        self.worker_thread = None
        self.lock = threading.RLock()
        
        # Event callbacks
        self.event_callbacks = []
        
        logger.info(f"TaskManager initialized with {max_workers} workers, "
                  f"{'process' if use_processes else 'thread'} executor")
        
    def add_task(self, name: str, func: Callable, args: Tuple = None, 
                 kwargs: Dict = None, priority: int = 0, retries: int = None, 
                 timeout: int = None, dependencies: List[str] = None) -> str:
        """
        Add a task to the manager.
        
        Args:
            name: Human-readable name for the task
            func: The function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Task priority (higher values = higher priority)
            retries: Number of retry attempts if the task fails, or None for default
            timeout: Timeout in seconds, or None for default
            dependencies: List of task IDs that must complete before this task
            
        Returns:
            task_id: The unique ID assigned to the task
        """
        task_id = str(uuid.uuid4())
        
        with self.lock:
            task = Task(
                task_id=task_id,
                name=name,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                retries=retries if retries is not None else self.max_retries,
                timeout=timeout if timeout is not None else self.default_timeout,
                dependencies=dependencies
            )
            
            self.tasks[task_id] = task
            
            # Check if we can enqueue immediately or need to wait for dependencies
            if not dependencies:
                self.task_queue.put(task)
                logger.debug(f"Added task {task_id} ({name}) to queue with priority {priority}")
            else:
                # Check for missing dependencies
                missing_deps = [dep for dep in dependencies if dep not in self.tasks]
                if missing_deps:
                    logger.warning(f"Task {task_id} ({name}) has missing dependencies: {missing_deps}")
                
                logger.debug(f"Added task {task_id} ({name}) with dependencies {dependencies}")
            
            # Notify event listeners
            self._notify_event("task_added", task_id)
            
            # Start the worker thread if it's not running
            if not self.running:
                self.start()
                
        return task_id
        
    def add_tasks(self, tasks: List[Dict]) -> List[str]:
        """
        Add multiple tasks at once.
        
        Args:
            tasks: List of task dictionaries with the same parameters as add_task
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        for task_dict in tasks:
            task_id = self.add_task(**task_dict)
            task_ids.append(task_id)
            
        return task_ids
        
    def get_task(self, task_id: str) -> Optional[Dict]:
        """
        Get task information by ID.
        
        Args:
            task_id: The task ID to look up
            
        Returns:
            Task information dictionary or None if not found
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                return task.to_dict()
            return None
            
    def get_all_tasks(self) -> List[Dict]:
        """
        Get information about all tasks.
        
        Returns:
            List of task information dictionaries
        """
        with self.lock:
            return [task.to_dict() for task in self.tasks.values()]
            
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it hasn't started yet.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if the task was canceled, False otherwise
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
                
            if task.status == "pending":
                task.status = "cancelled"
                self._notify_event("task_cancelled", task_id)
                logger.info(f"Cancelled task {task_id} ({task.name})")
                return True
                
            if task.status == "running" and task.future:
                cancelled = task.future.cancel()
                if cancelled:
                    task.status = "cancelled"
                    self.running_tasks.remove(task_id)
                    self._notify_event("task_cancelled", task_id)
                    logger.info(f"Cancelled running task {task_id} ({task.name})")
                    return True
                    
            return False
            
    def update_task_progress(self, task_id: str, progress: float, message: str = None) -> bool:
        """
        Update the progress of a task.
        
        Args:
            task_id: ID of the task to update
            progress: Progress value between 0.0 and 1.0
            message: Optional message about the progress
            
        Returns:
            True if the update was successful, False otherwise
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
                
            # Clamp progress to valid range
            progress = max(0.0, min(1.0, progress))
            
            # Update progress
            task.progress = progress
            
            # Log the message if provided
            if message:
                timestamp = time.time()
                task.logs.append((timestamp, message))
                
            # Notify listeners
            self._notify_event("task_progress", task_id, progress=progress, message=message)
            
            return True
            
    def get_task_result(self, task_id: str) -> Dict:
        """
        Get the result of a completed task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with task result information
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return {"success": False, "error": "Task not found"}
                
            if task.status == "completed":
                return {
                    "success": True,
                    "result": task.result,
                    "task_id": task_id,
                    "execution_time": (task.end_time - task.start_time) if task.end_time and task.start_time else None
                }
            elif task.status == "failed":
                return {
                    "success": False,
                    "error": str(task.error),
                    "task_id": task_id,
                    "attempts": task.attempts
                }
            else:
                return {
                    "success": False,
                    "error": f"Task is {task.status}, not completed",
                    "task_id": task_id,
                    "status": task.status,
                    "progress": task.progress
                }
                
    def add_event_listener(self, callback: Callable[[str, str, Dict], None]) -> None:
        """
        Add an event listener for task events.
        
        Args:
            callback: Function that takes (event_type, task_id, event_data)
        """
        with self.lock:
            self.event_callbacks.append(callback)
            
    def remove_event_listener(self, callback: Callable) -> bool:
        """
        Remove an event listener.
        
        Args:
            callback: The callback function to remove
            
        Returns:
            True if the callback was removed, False if not found
        """
        with self.lock:
            try:
                self.event_callbacks.remove(callback)
                return True
            except ValueError:
                return False
                
    def _notify_event(self, event_type: str, task_id: str, **kwargs) -> None:
        """Send event notification to all listeners."""
        event_data = {"task_id": task_id, **kwargs}
        
        # Copy the callbacks to avoid issues if the list changes during iteration
        callbacks = list(self.event_callbacks)
        
        # Call each callback without holding the lock
        for callback in callbacks:
            try:
                callback(event_type, task_id, event_data)
            except Exception as e:
                logger.error(f"Error in event listener: {str(e)}")
                
    def _init_executor(self):
        """Initialize the executor if it doesn't exist yet."""
        if self.executor is None:
            if self.use_processes:
                self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                
    def _check_dependencies(self, task: Task) -> bool:
        """
        Check if all dependencies of a task are completed.
        
        Args:
            task: The task to check
            
        Returns:
            True if all dependencies are complete or there are no dependencies
        """
        if not task.dependencies:
            return True
            
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != "completed":
                return False
                
        return True
        
    def _worker_loop(self) -> None:
        """Main worker loop that processes the task queue."""
        logger.info("Task manager worker thread started")
        
        # Initialize the executor
        self._init_executor()
        
        while self.running:
            try:
                # Check for completed futures
                self._process_completed_futures()
                
                # Process dependency chain - check if any pending tasks can now run
                self._check_dependency_chain()
                
                # Get the next task if we have capacity
                if len(self.running_tasks) < self.max_workers:
                    try:
                        # Non-blocking get with timeout
                        task = self.task_queue.get(block=True, timeout=0.1)
                        
                        # Ignore cancelled tasks
                        if task.status == "cancelled":
                            self.task_queue.task_done()
                            continue
                            
                        # Execute the task
                        self._execute_task(task)
                        
                    except queue.Empty:
                        # No tasks in queue, just continue
                        pass
                else:
                    # All workers busy, wait a bit
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in task manager worker loop: {str(e)}")
                time.sleep(1)  # Prevent high CPU usage in case of repeated errors
                
        logger.info("Task manager worker thread stopped")
        
    def _execute_task(self, task: Task) -> None:
        """
        Submit a task to the executor.
        
        Args:
            task: The task to execute
        """
        # Update task status
        with self.lock:
            task.status = "running"
            task.start_time = time.time()
            task.attempts += 1
            self.running_tasks.add(task.task_id)
            
        # Create a progress callback
        progress_cb = TaskProgressCallback(task.task_id, self)
        
        # Add progress_callback to kwargs if the function accepts it
        kwargs = task.kwargs.copy()
        if 'progress_callback' not in kwargs:
            kwargs['progress_callback'] = progress_cb
            
        # Submit the task to the executor
        future = self.executor.submit(
            self._wrapped_task_func,
            task.func, task.args, kwargs, task.task_id, task.timeout
        )
        
        # Store the future for later checking
        task.future = future
        
        # Log and notify
        logger.info(f"Started task {task.task_id} ({task.name}), attempt {task.attempts}/{task.retries+1}")
        self._notify_event("task_started", task.task_id, attempt=task.attempts)
        
    def _wrapped_task_func(self, func, args, kwargs, task_id, timeout):
        """
        Wrapper around the task function to handle timeouts and exceptions.
        
        Args:
            func: The function to call
            args: Positional arguments
            kwargs: Keyword arguments
            task_id: Task ID for error reporting
            timeout: Timeout in seconds
            
        Returns:
            The result of the function call
        """
        # Handle progress_callback if the function doesn't accept it
        if 'progress_callback' in kwargs:
            try:
                # Try calling with it first (most tasks should accept it)
                return func(*args, **kwargs)
            except TypeError as e:
                if "unexpected keyword argument" in str(e) and "progress_callback" in str(e):
                    # Function doesn't accept progress_callback, remove it and try again
                    kwargs = kwargs.copy()
                    del kwargs['progress_callback']
                    return func(*args, **kwargs)
                else:
                    # Some other TypeError, re-raise
                    raise
        else:
            # No progress_callback, just call the function
            return func(*args, **kwargs)
            
    def _process_completed_futures(self) -> None:
        """Check for and process completed task futures."""
        # Make a copy of running_tasks to avoid modifying during iteration
        with self.lock:
            running_task_ids = list(self.running_tasks)
            
        for task_id in running_task_ids:
            with self.lock:
                task = self.tasks.get(task_id)
                if not task or not task.future:
                    continue
                    
                # Skip if the future isn't done yet
                if not task.future.done():
                    continue
                    
                # Get the result or exception
                try:
                    result = task.future.result(timeout=0)  # Non-blocking check
                    self._handle_task_success(task, result)
                except Exception as e:
                    self._handle_task_failure(task, e)
                    
    def _handle_task_success(self, task: Task, result: Any) -> None:
        """
        Handle successful task completion.
        
        Args:
            task: The completed task
            result: The task result
        """
        with self.lock:
            task.status = "completed"
            task.end_time = time.time()
            task.result = result
            task.progress = 1.0
            
            # Update tracking sets
            self.running_tasks.remove(task.task_id)
            self.completed_tasks.add(task.task_id)
            
            # Log and notify
            elapsed = task.end_time - task.start_time
            logger.info(f"Task {task.task_id} ({task.name}) completed successfully in {elapsed:.2f}s")
            self._notify_event("task_completed", task.task_id, elapsed=elapsed)
            
            # Mark queue item as done if it was executed directly from queue
            try:
                self.task_queue.task_done()
            except ValueError:
                # Task wasn't in queue (e.g., it was added later due to dependencies)
                pass
                
    def _handle_task_failure(self, task: Task, error: Exception) -> None:
        """
        Handle task failure, including retries if applicable.
        
        Args:
            task: The failed task
            error: The exception raised
        """
        with self.lock:
            # Update tracking sets
            self.running_tasks.remove(task.task_id)
            
            # Check if we should retry
            if task.attempts <= task.retries:
                # Reset status for retry
                task.status = "pending"
                task.error = error
                task.future = None
                
                # Log the retry
                logger.warning(f"Task {task.task_id} ({task.name}) failed, will retry "
                             f"(attempt {task.attempts}/{task.retries+1}): {str(error)}")
                             
                # Re-queue the task
                self.task_queue.put(task)
                self._notify_event("task_retry", task.task_id, 
                                 error=str(error), 
                                 attempt=task.attempts)
                
            else:
                # Max retries reached, mark as failed
                task.status = "failed"
                task.end_time = time.time()
                task.error = error
                self.failed_tasks.add(task.task_id)
                
                # Log the failure
                logger.error(f"Task {task.task_id} ({task.name}) failed after "
                           f"{task.attempts} attempts: {str(error)}")
                           
                # Notify listeners
                self._notify_event("task_failed", task.task_id, 
                                 error=str(error), 
                                 attempts=task.attempts)
                
            # Mark queue item as done
            try:
                self.task_queue.task_done()
            except ValueError:
                # Task wasn't in queue (e.g., it was a dependency)
                pass
                
    def _check_dependency_chain(self) -> None:
        """Check all tasks and queue those whose dependencies are met."""
        with self.lock:
            for task_id, task in self.tasks.items():
                # Only check pending tasks that aren't already in the queue
                if task.status == "pending" and task.dependencies and task not in self.task_queue.queue:
                    if self._check_dependencies(task):
                        # All dependencies satisfied, queue the task
                        self.task_queue.put(task)
                        logger.debug(f"Dependencies met for task {task_id} ({task.name}), adding to queue")
                        
    def start(self) -> None:
        """Start the task manager worker thread."""
        with self.lock:
            if not self.running:
                self.running = True
                self.worker_thread = threading.Thread(
                    target=self._worker_loop,
                    daemon=True
                )
                self.worker_thread.start()
                logger.info("Task manager started")
                
    def stop(self, wait: bool = True, timeout: float = 10.0) -> None:
        """
        Stop the task manager.
        
        Args:
            wait: If True, wait for running tasks to complete
            timeout: Maximum time to wait for tasks to complete
        """
        with self.lock:
            self.running = False
            
        if wait and self.worker_thread:
            self.worker_thread.join(timeout)
            
        # Shutdown the executor
        if self.executor:
            self.executor.shutdown(wait=wait, cancel_futures=not wait)
            self.executor = None
            
        logger.info("Task manager stopped")
        
    def wait_for_tasks(self, task_ids: List[str] = None, timeout: float = None) -> Dict[str, str]:
        """
        Wait for specific tasks or all tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for, or None for all tasks
            timeout: Maximum time to wait in seconds, or None to wait indefinitely
            
        Returns:
            Dictionary mapping task IDs to their final status
        """
        start_time = time.time()
        result = {}
        
        # If no task_ids provided, wait for all tasks
        if task_ids is None:
            with self.lock:
                task_ids = list(self.tasks.keys())
                
        # Wait for each task
        remaining_ids = set(task_ids)
        while remaining_ids and (timeout is None or time.time() - start_time < timeout):
            with self.lock:
                for task_id in list(remaining_ids):
                    task = self.tasks.get(task_id)
                    if not task:
                        # Task not found
                        result[task_id] = "not_found"
                        remaining_ids.remove(task_id)
                    elif task.status in ["completed", "failed", "cancelled"]:
                        # Task finished
                        result[task_id] = task.status
                        remaining_ids.remove(task_id)
                        
            # If we still have tasks to wait for, sleep briefly
            if remaining_ids:
                time.sleep(0.1)
                
        # For any remaining tasks, mark as "timeout"
        for task_id in remaining_ids:
            result[task_id] = "timeout"
            
        return result
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop(wait=True)
        
        
# Convenience functions for creating tasks without directly using TaskManager

_default_task_manager = None

def get_default_task_manager() -> TaskManager:
    """Get the default task manager instance, creating it if needed."""
    global _default_task_manager
    if _default_task_manager is None:
        _default_task_manager = TaskManager()
        _default_task_manager.start()
    return _default_task_manager
    
def submit_task(name: str, func: Callable, *args, **kwargs) -> str:
    """
    Submit a task to the default task manager.
    
    Args:
        name: Human-readable name for the task
        func: The function to execute
        *args: Positional arguments for the function
        **kwargs: Special kwargs for task submission are:
                  - priority: Task priority (default 0)
                  - retries: Max retry attempts (default from task manager)
                  - timeout: Timeout in seconds (default from task manager)
                  - dependencies: List of dependent task IDs
                  All other kwargs are passed to the function.
                  
    Returns:
        task_id: The unique ID assigned to the task
    """
    # Extract special kwargs
    task_kwargs = {}
    func_kwargs = {}
    
    for key, value in kwargs.items():
        if key in ['priority', 'retries', 'timeout', 'dependencies']:
            task_kwargs[key] = value
        else:
            func_kwargs[key] = value
            
    # Get the default task manager
    task_manager = get_default_task_manager()
    
    # Submit the task
    return task_manager.add_task(
        name=name,
        func=func,
        args=args,
        kwargs=func_kwargs,
        **task_kwargs
    )
    
def wait_for_task(task_id: str, timeout: float = None) -> Dict:
    """
    Wait for a task to complete and get its result.
    
    Args:
        task_id: The task ID to wait for
        timeout: Maximum time to wait in seconds
        
    Returns:
        Task result dictionary
    """
    task_manager = get_default_task_manager()
    
    # Wait for the task
    statuses = task_manager.wait_for_tasks([task_id], timeout)
    
    # Get the result
    return task_manager.get_task_result(task_id)
    
def execute_parallel(tasks: List[Dict]) -> Dict[str, Dict]:
    """
    Execute multiple tasks in parallel and wait for all to complete.
    
    Args:
        tasks: List of task definitions (each with name, func, args, kwargs)
        
    Returns:
        Dictionary mapping task IDs to their results
    """
    task_manager = get_default_task_manager()
    
    # Submit all tasks
    task_ids = task_manager.add_tasks(tasks)
    
    # Wait for all tasks to complete
    task_manager.wait_for_tasks(task_ids)
    
    # Get results
    results = {}
    for task_id in task_ids:
        results[task_id] = task_manager.get_task_result(task_id)
        
    return results


# Run a quick demo if this module is run directly
if __name__ == "__main__":
    import random
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s"
    )
    
    def test_task(n, sleep_time=1, progress_callback=None):
        """Test task that just sleeps and returns a value."""
        total_sleep = sleep_time
        
        # Report progress in chunks if callback provided
        if progress_callback:
            steps = 10
            for i in range(steps):
                time.sleep(sleep_time / steps)
                progress_callback.update((i+1)/steps, f"Step {i+1}/{steps}")
        else:
            time.sleep(sleep_time)
            
        return f"Task {n} completed after {total_sleep}s"
        
    def failing_task(fail_chance=0.5):
        """Task that randomly fails."""
        if random.random() < fail_chance:
            raise ValueError("Random failure!")
        return "Task succeeded!"
        
    # Create a task manager
    manager = TaskManager(max_workers=4)
    
    # Add a simple event listener
    def event_listener(event_type, task_id, event_data):
        logger.info(f"Event: {event_type} - Task: {task_id} - Data: {event_data}")
        
    manager.add_event_listener(event_listener)
    
    # Start the manager
    manager.start()
    
    try:
        # Add some tasks
        logger.info("Adding tasks...")
        
        # Simple tasks
        for i in range(5):
            task_id = manager.add_task(
                name=f"Simple Task {i}",
                func=test_task,
                args=(i, random.uniform(1, 3)),
                priority=i
            )
            logger.info(f"Added task {task_id}")
            
        # Task with dependency chain
        task1 = manager.add_task(
            name="Chain Task 1",
            func=test_task,
            args=(101, 2),
            priority=10
        )
        
        task2 = manager.add_task(
            name="Chain Task 2",
            func=test_task,
            args=(102, 1),
            dependencies=[task1]
        )
        
        task3 = manager.add_task(
            name="Chain Task 3",
            func=test_task,
            args=(103, 1),
            dependencies=[task2]
        )
        
        # Task that will fail and retry
        fail_task = manager.add_task(
            name="Failing Task",
            func=failing_task,
            args=(0.7,),  # 70% chance to fail
            retries=3
        )
        
        # Wait for all tasks to complete
        logger.info("Waiting for tasks to complete...")
        time.sleep(10)
        
        # Get results
        for task_id in [task1, task2, task3, fail_task]:
            result = manager.get_task_result(task_id)
            logger.info(f"Result for task {task_id}: {result}")
            
    finally:
        # Stop the manager
        logger.info("Stopping task manager...")
        manager.stop()
        logger.info("Done.")