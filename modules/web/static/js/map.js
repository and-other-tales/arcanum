/**
 * Arcanum Web Interface - Map Handler
 * 
 * This file contains functionality for the interactive map selection.
 */

// Global variables
let map = null;
let drawingRect = null;
let rectangle = null;
let geocoder = null;

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize map
    initializeMap();
    
    // Setup map-related event listeners
    setupMapEventListeners();
});

/**
 * Initialize the Leaflet map
 */
function initializeMap() {
    // Create map
    map = L.map('mapContainer').setView([51.505, -0.09], 13);
    
    // Add base layer (OpenStreetMap)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(map);
    
    // Add scale control
    L.control.scale().addTo(map);
    
    // Initialize drawing controls
    initializeDrawing();
}

/**
 * Initialize drawing rectangle functionality
 */
function initializeDrawing() {
    // Create a button to toggle drawing mode
    const drawControl = L.control({position: 'topleft'});
    
    drawControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
        div.innerHTML = '<a href="#" title="Draw Area" id="drawToggleBtn"><i class="fas fa-draw-polygon"></i></a>';
        return div;
    };
    
    drawControl.addTo(map);
    
    // Toggle drawing mode
    document.getElementById('drawToggleBtn').addEventListener('click', function(e) {
        e.preventDefault();
        
        if (drawingRect) {
            // Disable drawing mode
            map.off('mousedown', startDrawing);
            map.off('mousemove', updateRectangle);
            map.off('mouseup', finishDrawing);
            
            drawingRect = false;
            this.classList.remove('active');
            map.dragging.enable();
        } else {
            // Enable drawing mode
            map.on('mousedown', startDrawing);
            map.on('mouseup', finishDrawing);
            
            drawingRect = true;
            this.classList.add('active');
            map.dragging.disable();
        }
    });
}

/**
 * Start drawing rectangle
 * @param {Event} e - Mouse event
 */
function startDrawing(e) {
    // Clear existing rectangle
    if (rectangle) {
        map.removeLayer(rectangle);
    }
    
    // Get start coordinates
    const start = e.latlng;
    
    // Create rectangle with zero size
    rectangle = L.rectangle([
        [start.lat, start.lng],
        [start.lat, start.lng]
    ], {
        color: "#3388ff",
        weight: 3,
        opacity: 0.6,
        fillOpacity: 0.2
    }).addTo(map);
    
    // Enable mousemove handler
    map.on('mousemove', updateRectangle);
}

/**
 * Update rectangle size during drawing
 * @param {Event} e - Mouse event
 */
function updateRectangle(e) {
    if (rectangle) {
        // Get the original bounds
        const bounds = rectangle.getBounds();
        const nw = bounds.getNorthWest();
        
        // Update with current mouse position
        rectangle.setBounds([
            [nw.lat, nw.lng],
            [e.latlng.lat, e.latlng.lng]
        ]);
    }
}

/**
 * Finish drawing rectangle
 * @param {Event} e - Mouse event
 */
function finishDrawing(e) {
    // Disable mousemove handler
    map.off('mousemove', updateRectangle);
    
    if (rectangle) {
        // Get final bounds
        const bounds = rectangle.getBounds();
        
        // Update bounds inputs
        document.getElementById('boundNorth').value = bounds.getNorth().toFixed(6);
        document.getElementById('boundSouth').value = bounds.getSouth().toFixed(6);
        document.getElementById('boundEast').value = bounds.getEast().toFixed(6);
        document.getElementById('boundWest').value = bounds.getWest().toFixed(6);
        
        // Store bounds for generation
        selectedBounds = {
            north: bounds.getNorth(),
            south: bounds.getSouth(),
            east: bounds.getEast(),
            west: bounds.getWest()
        };
    }
}

/**
 * Setup map-related event listeners
 */
function setupMapEventListeners() {
    // Location search button
    document.getElementById('searchBtn').addEventListener('click', searchLocation);
    
    // Search on Enter key
    document.getElementById('locationSearch').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchLocation();
        }
    });
}

/**
 * Search for a location
 */
function searchLocation() {
    const locationInput = document.getElementById('locationSearch').value;
    
    if (!locationInput) {
        return;
    }
    
    // Show loading indicator
    const searchBtn = document.getElementById('searchBtn');
    const originalContent = searchBtn.innerHTML;
    searchBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
    searchBtn.disabled = true;
    
    // Use Nominatim for geocoding (free and open-source)
    fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(locationInput)}`)
        .then(response => response.json())
        .then(data => {
            // Reset button
            searchBtn.innerHTML = originalContent;
            searchBtn.disabled = false;
            
            if (data && data.length > 0) {
                // Get first result
                const result = data[0];
                
                // Move map to location
                map.setView([result.lat, result.lon], 15);
                
                // Create a marker
                const marker = L.marker([result.lat, result.lon]).addTo(map);
                marker.bindPopup(`<b>${result.display_name}</b>`).openPopup();
                
                // Auto-remove marker after 5 seconds
                setTimeout(() => {
                    map.removeLayer(marker);
                }, 5000);
                
                // Create a default rectangle around the point
                if (rectangle) {
                    map.removeLayer(rectangle);
                }
                
                // Create rectangle (approximately 500m x 500m)
                const lat = parseFloat(result.lat);
                const lon = parseFloat(result.lon);
                const latOffset = 0.0025; // About 250m in lat direction
                const lonOffset = 0.0035; // About 250m in lon direction (varies with latitude)
                
                rectangle = L.rectangle([
                    [lat + latOffset, lon - lonOffset],
                    [lat - latOffset, lon + lonOffset]
                ], {
                    color: "#3388ff",
                    weight: 3,
                    opacity: 0.6,
                    fillOpacity: 0.2
                }).addTo(map);
                
                // Update bounds inputs and store for generation
                const bounds = rectangle.getBounds();
                document.getElementById('boundNorth').value = bounds.getNorth().toFixed(6);
                document.getElementById('boundSouth').value = bounds.getSouth().toFixed(6);
                document.getElementById('boundEast').value = bounds.getEast().toFixed(6);
                document.getElementById('boundWest').value = bounds.getWest().toFixed(6);
                
                selectedBounds = {
                    north: bounds.getNorth(),
                    south: bounds.getSouth(),
                    east: bounds.getEast(),
                    west: bounds.getWest()
                };
            } else {
                // Show error
                showNotification('Location Not Found', 'No results found for your search. Try a different location or be more specific.', 'error');
            }
        })
        .catch(error => {
            // Reset button
            searchBtn.innerHTML = originalContent;
            searchBtn.disabled = false;
            
            console.error('Error searching location:', error);
            showNotification('Error', 'An error occurred while searching for the location.', 'error');
        });
}

/**
 * Show bounds on a map
 * @param {Object} bounds - Bounds object with north, south, east, west properties
 * @param {L.Map} targetMap - Target Leaflet map
 */
function showBoundsOnMap(bounds, targetMap) {
    // Create rectangle from bounds
    const rect = L.rectangle([
        [bounds.north, bounds.west],
        [bounds.south, bounds.east]
    ], {
        color: "#3388ff",
        weight: 3,
        opacity: 0.6,
        fillOpacity: 0.2
    }).addTo(targetMap);
    
    // Fit map to rectangle
    targetMap.fitBounds(rect.getBounds());
}