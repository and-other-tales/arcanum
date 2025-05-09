/**
 * Application configuration
 */
require('dotenv').config();

module.exports = {
  server: {
    port: process.env.PORT || 3000,
    env: process.env.NODE_ENV || 'development',
  },
  weather: {
    apiKey: process.env.WEATHER_API_KEY,
    baseUrl: process.env.WEATHER_API_BASE_URL || 'https://api.weatherapi.com/v1',
    requestInterval: parseInt(process.env.WEATHER_REQUEST_INTERVAL || 300000, 10), // Default to 5 minutes
  },
  london: {
    centerLat: parseFloat(process.env.LONDON_CENTER_LAT || 51.509865),
    centerLon: parseFloat(process.env.LONDON_CENTER_LON || -0.118092),
    // London districts with their coordinates
    districts: {
      westminster: { lat: 51.4975, lon: -0.1357 },
      city: { lat: 51.5155, lon: -0.0922 },
      southwark: { lat: 51.5031, lon: -0.0876 },
      lambeth: { lat: 51.4963, lon: -0.1168 },
      camden: { lat: 51.5390, lon: -0.1425 },
      islington: { lat: 51.5416, lon: -0.1028 },
      hackney: { lat: 51.5450, lon: -0.0553 },
      tower_hamlets: { lat: 51.5078, lon: -0.0300 },
    },
  },
  websocket: {
    port: process.env.WEBSOCKET_PORT || 3001,
  },
  time: {
    updateInterval: parseInt(process.env.TIME_UPDATE_INTERVAL || 60000, 10), // Default to 60 seconds
  },
};
