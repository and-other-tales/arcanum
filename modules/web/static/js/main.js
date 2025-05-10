/**
 * Arcanum City Generator - Web Interface
 * Main JavaScript for the web interface
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to clicked link
            this.classList.add('active');
            
            // Get target tab
            const targetId = this.getAttribute('data-bs-target');
            const targetTab = document.querySelector(targetId);
            
            // Hide all tabs
            document.querySelectorAll('.tab-pane').forEach(tab => {
                tab.classList.remove('show', 'active');
            });
            
            // Show target tab
            if (targetTab) {
                targetTab.classList.add('show', 'active');
                
                // Load tab-specific content
                if (targetId === '#textureManagerTab') {
                    loadTextureManagerData();
                }
            }
        });
    });
    
    // Activate default tab (first one)
    if (navLinks.length > 0) {
        navLinks[0].click();
    }
    
    // Initialize texture manager UI events
    initTextureManager();
});

// Texture Manager Functions
let buildings = {};
let textures = [];
let atlases = [];
let selectedBuildingId = null;

function initTextureManager() {
    // Building search functionality
    const buildingSearchInput = document.getElementById('buildingSearch');
    if (buildingSearchInput) {
        buildingSearchInput.addEventListener('input', function() {
            filterBuildings(this.value.trim().toLowerCase());
        });
    }
    
    // Atlas selector change event
    const atlasSelect = document.getElementById('atlasSelect');
    if (atlasSelect) {
        atlasSelect.addEventListener('change', function() {
            if (this.value) {
                loadAtlasPreview(this.value);
            } else {
                // Clear atlas preview
                document.getElementById('atlasPreviewContainer').innerHTML = `
                    <div class="text-center py-5 text-muted">
                        <i class="fas fa-th fa-2x mb-3"></i>
                        <p>Select an atlas to view preview</p>
                    </div>
                `;
            }
        });
    }
    
    // Assign texture button click event
    const assignTextureBtn = document.getElementById('assignTextureBtn');
    if (assignTextureBtn) {
        assignTextureBtn.addEventListener('click', function() {
            const textureSelect = document.getElementById('textureSelect');
            const selectedTextureId = textureSelect.value;
            
            if (selectedBuildingId && selectedTextureId) {
                assignTextureToBuilding(selectedBuildingId, selectedTextureId);
            }
        });
    }
    
    // Regenerate atlas button click event
    const regenerateAtlasBtn = document.getElementById('regenerateAtlasBtn');
    if (regenerateAtlasBtn) {
        regenerateAtlasBtn.addEventListener('click', function() {
            // TODO: Implement atlas regeneration
            alert('Atlas regeneration not implemented yet');
        });
    }
    
    // Download atlas button click event
    const downloadAtlasBtn = document.getElementById('downloadAtlasBtn');
    if (downloadAtlasBtn) {
        downloadAtlasBtn.addEventListener('click', function() {
            const atlasSelect = document.getElementById('atlasSelect');
            const selectedAtlasId = atlasSelect.value;
            
            if (selectedAtlasId) {
                const downloadUrl = `/api/atlases/${selectedAtlasId}`;
                window.open(downloadUrl, '_blank');
            }
        });
    }
}

function loadTextureManagerData() {
    // Load buildings, textures, and atlases in parallel
    Promise.all([
        fetch('/api/buildings').then(response => response.json()),
        fetch('/api/textures').then(response => response.json()),
        fetch('/api/atlases').then(response => response.json())
    ])
    .then(([buildingsData, texturesData, atlasesData]) => {
        // Process buildings
        buildings = buildingsData.buildings || {};
        populateBuildingsList(buildings);
        
        // Process textures
        textures = texturesData.textures || [];
        populateTextureSelect(textures);
        
        // Process atlases
        atlases = atlasesData.atlases || [];
        populateAtlasSelect(atlases);
    })
    .catch(error => {
        console.error('Error loading texture manager data:', error);
    });
}

function populateBuildingsList(buildings) {
    const buildingsList = document.getElementById('buildingsList');
    
    if (Object.keys(buildings).length === 0) {
        buildingsList.innerHTML = `
            <div class="text-center py-5 text-muted">
                <i class="fas fa-building fa-2x mb-3"></i>
                <p>No buildings found</p>
            </div>
        `;
        return;
    }
    
    // Create list items for each building
    let listItems = '';
    for (const [buildingId, building] of Object.entries(buildings)) {
        const buildingName = building.name || buildingId;
        const buildingType = building.type || 'standard';
        const hasTexture = building.texture_assigned ? 'has-texture' : '';
        
        listItems += `
            <a href="#" class="list-group-item list-group-item-action ${hasTexture}" data-building-id="${buildingId}">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${buildingName}</h6>
                    <small class="text-muted">${buildingType}</small>
                </div>
                <small>ID: ${buildingId}</small>
            </a>
        `;
    }
    
    buildingsList.innerHTML = listItems;
    
    // Add click event listeners
    buildingsList.querySelectorAll('.list-group-item').forEach(item => {
        item.addEventListener('click', function() {
            const buildingId = this.getAttribute('data-building-id');
            selectBuilding(buildingId);
            
            // Mark this item as active
            buildingsList.querySelectorAll('.list-group-item').forEach(i => {
                i.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
}

function populateTextureSelect(textures) {
    const textureSelect = document.getElementById('textureSelect');
    
    if (textures.length === 0) {
        textureSelect.innerHTML = '<option value="">No textures available</option>';
        textureSelect.disabled = true;
        return;
    }
    
    // Create options for each texture
    let options = '<option value="">Select a texture...</option>';
    for (const texture of textures) {
        options += `<option value="${texture.id}">${texture.name}</option>`;
    }
    
    textureSelect.innerHTML = options;
    textureSelect.disabled = false;
}

function populateAtlasSelect(atlases) {
    const atlasSelect = document.getElementById('atlasSelect');
    const regenerateAtlasBtn = document.getElementById('regenerateAtlasBtn');
    const downloadAtlasBtn = document.getElementById('downloadAtlasBtn');
    
    if (atlases.length === 0) {
        atlasSelect.innerHTML = '<option value="">No atlas available</option>';
        atlasSelect.disabled = true;
        regenerateAtlasBtn.disabled = true;
        downloadAtlasBtn.disabled = true;
        return;
    }
    
    // Create options for each atlas
    let options = '<option value="">Select an atlas...</option>';
    for (const atlas of atlases) {
        options += `<option value="${atlas.id}">${atlas.name}</option>`;
    }
    
    atlasSelect.innerHTML = options;
    atlasSelect.disabled = false;
    regenerateAtlasBtn.disabled = false;
    downloadAtlasBtn.disabled = false;
}

function selectBuilding(buildingId) {
    selectedBuildingId = buildingId;
    const building = buildings[buildingId];
    
    if (!building) {
        console.error(`Building not found: ${buildingId}`);
        return;
    }
    
    // Update building detail container
    const buildingDetailContainer = document.getElementById('buildingDetailContainer');
    buildingDetailContainer.innerHTML = `
        <h6>Building ID: ${buildingId}</h6>
        <div class="mb-3">
            <label class="form-label">Name</label>
            <input type="text" class="form-control" value="${building.name || buildingId}" readonly>
        </div>
        <div class="mb-3">
            <label class="form-label">Type</label>
            <input type="text" class="form-control" value="${building.type || 'standard'}" readonly>
        </div>
        <div class="mb-3">
            <label class="form-label">Height</label>
            <input type="text" class="form-control" value="${building.height || 'N/A'}" readonly>
        </div>
        <div class="mb-3">
            <label class="form-label">Area</label>
            <input type="text" class="form-control" value="${building.area || 'N/A'}" readonly>
        </div>
    `;
    
    // Enable texture assignment controls
    document.getElementById('textureSelect').disabled = false;
    document.getElementById('assignTextureBtn').disabled = false;
    
    // Update current texture display
    updateCurrentTextureDisplay(buildingId);
    
    // Load building preview and texture mapping
    loadBuildingPreview(buildingId);
}

function updateCurrentTextureDisplay(buildingId) {
    const currentTextureContainer = document.getElementById('currentTextureContainer');
    const building = buildings[buildingId];
    
    if (building && building.texture_id) {
        // Find texture info
        const texture = textures.find(t => t.id === building.texture_id);
        
        if (texture) {
            currentTextureContainer.innerHTML = `
                <h6>Current Texture</h6>
                <div class="text-center">
                    <img src="${texture.url}" class="img-fluid mb-2 rounded" style="max-height: 150px;">
                    <p class="mb-0">${texture.name}</p>
                </div>
            `;
        } else {
            currentTextureContainer.innerHTML = `
                <h6>Current Texture</h6>
                <div class="text-center py-3 bg-light rounded">
                    <i class="fas fa-image fa-2x text-muted"></i>
                    <p class="text-muted mt-2">Texture not found: ${building.texture_id}</p>
                </div>
            `;
        }
    } else {
        currentTextureContainer.innerHTML = `
            <h6>Current Texture</h6>
            <div class="text-center py-3 bg-light rounded">
                <i class="fas fa-image fa-2x text-muted"></i>
                <p class="text-muted mt-2">No texture assigned</p>
            </div>
        `;
    }
}

function assignTextureToBuilding(buildingId, textureId) {
    // Show loading indicator
    document.getElementById('assignTextureBtn').innerHTML = `
        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        Assigning...
    `;
    document.getElementById('assignTextureBtn').disabled = true;
    
    // Send request to assign texture
    fetch('/api/textures/assign', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            building_id: buildingId,
            texture_id: textureId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update building in our local cache
            if (buildings[buildingId]) {
                buildings[buildingId].texture_id = textureId;
                buildings[buildingId].texture_assigned = true;
            }
            
            // Update display
            updateCurrentTextureDisplay(buildingId);
            
            // Mark building as having texture in the list
            const buildingListItem = document.querySelector(`.list-group-item[data-building-id="${buildingId}"]`);
            if (buildingListItem) {
                buildingListItem.classList.add('has-texture');
            }
            
            // Refresh atlas preview
            const atlasSelect = document.getElementById('atlasSelect');
            if (atlasSelect.value) {
                loadAtlasPreview(atlasSelect.value);
            }
            
            // Load building preview with new texture
            loadBuildingPreview(buildingId);
        } else {
            console.error('Error assigning texture:', data.error);
            alert(`Error assigning texture: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error assigning texture:', error);
        alert(`Error assigning texture: ${error.message}`);
    })
    .finally(() => {
        // Reset button
        document.getElementById('assignTextureBtn').innerHTML = `
            <i class="fas fa-link me-2"></i>Assign Texture
        `;
        document.getElementById('assignTextureBtn').disabled = false;
    });
}

function loadAtlasPreview(atlasId) {
    // Show loading indicator
    document.getElementById('atlasPreviewContainer').innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Loading atlas preview...</p>
        </div>
    `;
    
    // Request preview generation
    fetch(`/api/preview/atlas?atlas_id=${encodeURIComponent(atlasId)}`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Load the preview image
            document.getElementById('atlasPreviewContainer').innerHTML = `
                <img src="${data.preview_url}" class="atlas-image" alt="Atlas Preview">
                <div class="atlas-overlay" id="atlasOverlay"></div>
            `;
        } else {
            console.error('Error generating atlas preview:', data.error);
            document.getElementById('atlasPreviewContainer').innerHTML = `
                <div class="text-center py-5 text-muted">
                    <i class="fas fa-exclamation-circle fa-2x mb-3"></i>
                    <p>Error generating preview: ${data.error}</p>
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Error loading atlas preview:', error);
        document.getElementById('atlasPreviewContainer').innerHTML = `
            <div class="text-center py-5 text-muted">
                <i class="fas fa-exclamation-circle fa-2x mb-3"></i>
                <p>Error loading preview: ${error.message}</p>
            </div>
        `;
    });
}

function loadBuildingPreview(buildingId) {
    // Show loading indicators
    document.getElementById('buildingPreviewContainer').innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Loading building preview...</p>
        </div>
    `;
    
    document.getElementById('textureMapPreviewContainer').innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Loading texture mapping...</p>
        </div>
    `;
    
    // Request building preview with texture mapping
    fetch(`/api/preview/building?building_id=${encodeURIComponent(buildingId)}`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Load the preview images
            document.getElementById('textureMapPreviewContainer').innerHTML = `
                <img src="${data.preview_url}" class="img-fluid rounded" alt="Texture Mapping Preview">
            `;
            
            // For now, use the same image for building preview
            // In a real implementation, this would be a 3D model viewer
            document.getElementById('buildingPreviewContainer').innerHTML = `
                <div class="text-center">
                    <img src="${data.preview_url}" class="img-fluid rounded" alt="Building Preview">
                </div>
            `;
        } else {
            // Show error message
            document.getElementById('textureMapPreviewContainer').innerHTML = `
                <div class="text-center py-5 text-muted">
                    <i class="fas fa-exclamation-circle fa-2x mb-3"></i>
                    <p>Error generating preview: ${data.error}</p>
                </div>
            `;
            
            document.getElementById('buildingPreviewContainer').innerHTML = `
                <div class="text-center py-5 text-muted">
                    <i class="fas fa-exclamation-circle fa-2x mb-3"></i>
                    <p>Error generating preview: ${data.error}</p>
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Error loading building preview:', error);
        document.getElementById('textureMapPreviewContainer').innerHTML = `
            <div class="text-center py-5 text-muted">
                <i class="fas fa-exclamation-circle fa-2x mb-3"></i>
                <p>Error loading preview: ${error.message}</p>
            </div>
        `;
        
        document.getElementById('buildingPreviewContainer').innerHTML = `
            <div class="text-center py-5 text-muted">
                <i class="fas fa-exclamation-circle fa-2x mb-3"></i>
                <p>Error loading preview: ${error.message}</p>
            </div>
        `;
    });
}

function filterBuildings(searchText) {
    const buildingListItems = document.querySelectorAll('#buildingsList .list-group-item');
    
    buildingListItems.forEach(item => {
        const buildingId = item.getAttribute('data-building-id');
        const building = buildings[buildingId];
        const buildingName = building ? (building.name || buildingId) : buildingId;
        
        // Check if building name or ID matches search text
        if (buildingName.toLowerCase().includes(searchText) || buildingId.toLowerCase().includes(searchText)) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    });
}