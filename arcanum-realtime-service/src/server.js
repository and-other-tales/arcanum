/**
 * Arcanum Realtime Service - Main Server
 */
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

const config = require('./config/config');
const logger = require('./utils/logger');
const { initializeWeatherService } = require('./services/weatherService');
const { initializeTimeService } = require('./services/timeService');

// Initialize Express app
const app = express();

// Middleware
app.use(helmet()); // Security headers
app.use(cors()); // Enable CORS for all routes
app.use(express.json()); // Parse JSON request body
app.use(morgan('combined')); // HTTP request logging

// Import routes
const weatherRoutes = require('./routes/weatherRoutes');
const timeRoutes = require('./routes/timeRoutes');

// Register routes
app.use('/api/weather', weatherRoutes);
app.use('/api/time', timeRoutes);

// Simple health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Create HTTP server
const server = http.createServer(app);

// Initialize Socket.io for real-time communication
const io = socketIo(server, {
  cors: {
    origin: '*', // Allow connections from any origin
    methods: ['GET', 'POST'],
  }
});

// Socket.io connection handler
io.on('connection', (socket) => {
  logger.info(`Client connected: ${socket.id}`);
  
  // Handle client disconnection
  socket.on('disconnect', () => {
    logger.info(`Client disconnected: ${socket.id}`);
  });
  
  // Handle location updates from Unity
  socket.on('updateLocation', (data) => {
    logger.debug(`Location update from ${socket.id}: ${JSON.stringify(data)}`);
    // The weather service will use this location to fetch weather data
    if (data && data.latitude && data.longitude) {
      socket.latitude = data.latitude;
      socket.longitude = data.longitude;
    }
  });
});

// Start the server
server.listen(config.server.port, () => {
  logger.info(`Server running in ${config.server.env} mode on port ${config.server.port}`);
  
  // Initialize the WebSocket server for Unity communication
  const wsServer = require('./services/websocketService');
  wsServer.initialize(config.websocket.port);
  
  // Initialize the weather service
  initializeWeatherService(io);
  
  // Initialize the time service
  initializeTimeService(io);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
  logger.error('Unhandled Promise Rejection:', err);
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  logger.error('Uncaught Exception:', err);
  // Exit with failure
  process.exit(1);
});

module.exports = { app, server };
