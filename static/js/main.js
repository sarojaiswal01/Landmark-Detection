// State Management
let activeInputMode = 'upload';
let selectedFile = null;
let webcamStream = null;
let isProcessing = false;
let mapInstance = null;
let mapMarker = null;
let mapTileLayers = {};
let currentMapStyle = 'street';
let activeVisualMode = 'original';
let activeMetaTab = 'history';
let predictionResults = null;
let mapMarkers = [];
let travelRouteLine = null;

// Page Lifecycle Initialization
document.addEventListener('DOMContentLoaded', () => {
    // Fade out page loader
    setTimeout(() => {
        const loader = document.getElementById('page-loader');
        if (loader) {
            loader.style.opacity = '0';
            loader.style.visibility = 'hidden';
            // Recalculate leaflet map size once loader is fully gone
            setTimeout(() => {
                if (mapInstance) {
                    mapInstance.invalidateSize();
                }
            }, 300);
        }
    }, 1000);

    // Initial system queries
    initMap();
    fetchModelInfo();
    loadHistory();
    setupDragAndDrop();
    
    // Track mouse coordinates globally for Vercel grid spotlight glow effects
    document.addEventListener('mousemove', (e) => {
        document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`);
        document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`);
    });

    // Initialize 3D rotating telemetry globe in the background
    initBackgroundGlobe();
});

// Setup Drag & Drop Handlers
// Setup Drag & Drop Handlers
function setupDragAndDrop() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');

    if (!dropzone || !fileInput) return;

    // Trigger click on browse
    dropzone.addEventListener('click', (e) => {
        if (e.target.className !== 'browse-link') {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            if (e.target.files.length > 1) {
                handleFileSelection(e.target.files);
            } else {
                handleFileSelection(e.target.files[0]);
            }
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            if (e.dataTransfer.files.length > 1) {
                handleFileSelection(e.dataTransfer.files);
            } else {
                handleFileSelection(e.dataTransfer.files[0]);
            }
        }
    });
}

// Handle selected file details
function handleFileSelection(fileOrFiles) {
    // Check if multiple files are selected
    if (Array.isArray(fileOrFiles) || (fileOrFiles instanceof FileList) || (fileOrFiles instanceof Array)) {
        const files = Array.from(fileOrFiles).filter(f => f.type.startsWith('image/'));
        if (files.length === 0) {
            alert('Please upload valid image files.');
            return;
        }
        
        selectedFile = files;
        
        // Show batch grid and hide single image preview
        const singleWrapper = document.getElementById('single-preview-wrapper');
        const batchContainer = document.getElementById('batch-preview-container');
        
        singleWrapper.style.display = 'none';
        batchContainer.style.display = 'grid';
        batchContainer.innerHTML = '';
        
        // Render thumbnail items in batch grid
        files.forEach((file, idx) => {
            const reader = new FileReader();
            const item = document.createElement('div');
            item.className = 'batch-preview-item status-pending';
            item.id = `batch-item-${idx}`;
            
            reader.onload = (e) => {
                item.innerHTML = `
                    <img src="${e.target.result}" alt="${file.name}">
                    <span class="batch-status-badge">Ready</span>
                `;
            };
            reader.readAsDataURL(file);
            batchContainer.appendChild(item);
        });
        
        // Update predict button label
        const predictBtn = document.getElementById('btn-predict');
        predictBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Classify ${files.length} Images`;
        
        // Switch view to preview panel
        document.getElementById('mode-upload').classList.remove('active');
        document.getElementById('preview-panel').style.display = 'block';
        return;
    }

    // Single file selected (original flow)
    const file = fileOrFiles;
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }
    
    selectedFile = file;
    
    // Toggle containers
    const singleWrapper = document.getElementById('single-preview-wrapper');
    const batchContainer = document.getElementById('batch-preview-container');
    
    singleWrapper.style.display = 'block';
    batchContainer.style.display = 'none';
    batchContainer.innerHTML = '';
    
    // Render image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('image-preview');
        preview.src = e.target.result;
        
        // Switch view to preview panel
        document.getElementById('mode-upload').classList.remove('active');
        document.getElementById('preview-panel').style.display = 'block';
    };
    reader.readAsDataURL(file);
    
    // Reset predict button label
    const predictBtn = document.getElementById('btn-predict');
    predictBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Classify Image`;
}

// Input Mode Switchers
function switchInputMode(mode) {
    if (activeInputMode === mode) return;
    
    activeInputMode = mode;
    
    // Toggle active tab buttons
    document.getElementById('tab-upload').classList.toggle('active', mode === 'upload');
    document.getElementById('tab-camera').classList.toggle('active', mode === 'camera');
    
    // Toggle active workspace templates
    if (mode === 'upload') {
        stopCamera();
        document.getElementById('mode-upload').classList.add('active');
        document.getElementById('mode-camera').classList.remove('active');
        if (selectedFile) {
            document.getElementById('preview-panel').style.display = 'block';
            document.getElementById('mode-upload').classList.remove('active');
        }
    } else {
        document.getElementById('mode-upload').classList.remove('active');
        document.getElementById('mode-camera').classList.add('active');
        document.getElementById('preview-panel').style.display = 'none';
        startCamera();
    }
}

// Webcam stream controls
async function startCamera() {
    const webcam = document.getElementById('webcam');
    const placeholder = document.getElementById('camera-placeholder');
    const controls = document.getElementById('camera-controls');
    
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: 640, height: 480 },
            audio: false
        });
        webcam.srcObject = webcamStream;
        webcam.style.display = 'block';
        placeholder.style.display = 'none';
        controls.style.display = 'flex';
    } catch (err) {
        console.error('Camera Access Error:', err);
        alert('Could not access camera. Please check permissions.');
        switchInputMode('upload');
    }
}

function stopCamera() {
    const webcam = document.getElementById('webcam');
    const placeholder = document.getElementById('camera-placeholder');
    const controls = document.getElementById('camera-controls');
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    webcam.srcObject = null;
    webcam.style.display = 'none';
    placeholder.style.display = 'flex';
    controls.style.display = 'none';
}

function capturePhoto() {
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('photo-canvas');
    const ctx = canvas.getContext('2d');
    
    if (!webcamStream) return;
    
    // Set canvas sizes equal to video stream size
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    
    // Draw current frame on canvas
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    
    // Get Base64 image
    const dataUrl = canvas.toDataURL('image/jpeg');
    
    // Render preview
    const preview = document.getElementById('image-preview');
    preview.src = dataUrl;
    
    // Save base64 as file reference
    selectedFile = dataUrl;
    
    // Stop camera and switch panels
    stopCamera();
    document.getElementById('mode-camera').classList.remove('active');
    document.getElementById('preview-panel').style.display = 'block';
}

// Clear selected preview state
function resetPreview() {
    selectedFile = null;
    document.getElementById('preview-panel').style.display = 'none';
    document.getElementById('image-preview').src = '#';
    
    const singleWrapper = document.getElementById('single-preview-wrapper');
    const batchContainer = document.getElementById('batch-preview-container');
    singleWrapper.style.display = 'block';
    batchContainer.style.display = 'none';
    batchContainer.innerHTML = '';
    
    // Reset predict button label
    const predictBtn = document.getElementById('btn-predict');
    predictBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Classify Image`;
    
    if (activeInputMode === 'upload') {
        document.getElementById('mode-upload').classList.add('active');
    } else {
        document.getElementById('mode-camera').classList.add('active');
        startCamera();
    }
}

// System API Callers
async function fetchModelInfo() {
    try {
        const response = await fetch('/model-info');
        const data = await response.json();
        
        // Update header status pill
        const statusPill = document.getElementById('system-status');
        const statusLabel = statusPill.querySelector('.status-label');
        const statusDot = statusPill.querySelector('.status-dot');
        
        statusLabel.textContent = `Model: ${data.status}`;
        
        if (data.status === 'Emulated') {
            statusPill.classList.add('active');
            statusDot.style.backgroundColor = 'var(--accent-secondary)';
            statusDot.style.boxShadow = '0 0 8px var(--accent-secondary)';
        } else {
            statusPill.classList.toggle('active', data.model_loaded);
            statusDot.style.backgroundColor = '';
            statusDot.style.boxShadow = '';
        }
        
        // Update stats card details
        document.getElementById('stat-model-type').textContent = data.model_type;
        document.getElementById('stat-classes-count').textContent = `${data.classes_count} Landmarks`;
        document.getElementById('stat-threshold').textContent = `${Math.round(data.confidence_threshold * 100)}% Confidence`;
    } catch (err) {
        console.error('Error fetching model stats:', err);
    }
}

async function loadHistory(activeLat = null, activeLng = null) {
    try {
        const response = await fetch('/history');
        const history = await response.json();
        
        const emptyState = document.getElementById('history-empty');
        const grid = document.getElementById('history-grid');
        
        if (history.length === 0) {
            emptyState.style.display = 'block';
            grid.style.display = 'none';
            return;
        }
        
        emptyState.style.display = 'none';
        grid.style.display = 'grid';
        grid.innerHTML = '';
        
        history.forEach(item => {
            const card = document.createElement('div');
            card.className = 'history-card';
            
            // Build card html content
            const confidencePct = `${Math.round(item.confidence * 100)}%`;
            card.innerHTML = `
                <div class="history-img-box">
                    <img src="${item.image_url}" alt="${item.landmark_name}">
                </div>
                <div class="history-info">
                    <span class="history-name">${item.landmark_name}</span>
                    <span class="history-conf">${item.is_detected ? confidencePct : 'Unknown'}</span>
                    <span class="history-time">${item.timestamp}</span>
                </div>
            `;
            
            // Click listener to center map on coordinates
            if (item.latitude !== undefined && item.latitude !== null &&
                item.longitude !== undefined && item.longitude !== null) {
                card.style.cursor = 'pointer';
                card.addEventListener('click', () => {
                    if (mapInstance) {
                        mapInstance.flyTo([item.latitude, item.longitude], 13);
                        
                        // Trigger saved weather overlay effects on map
                        updateMeteorologicalOverlay(item.weather);
                        
                        // Find matching marker and open its popup
                        const marker = mapMarkers.find(m => {
                            const ll = m.getLatLng();
                            return Math.abs(ll.lat - item.latitude) < 0.0001 &&
                                   Math.abs(ll.lng - item.longitude) < 0.0001;
                        });
                        if (marker) {
                            marker.openPopup();
                        }
                    }
                });
            }
            
            grid.appendChild(card);
        });
        
        plotHistoryMarkers(history, activeLat, activeLng);
    } catch (err) {
        console.error('Error loading history:', err);
    }
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear your prediction history?')) return;
    try {
        await fetch('/clear-history', { method: 'POST' });
        loadHistory();
    } catch (err) {
        console.error('Error clearing history:', err);
    }
}

// Sample Gallery click loader
async function loadSample(landmarkKey) {
    // We map landmarkKeys to high quality Unsplash photos.
    // Instead of using simple placeholder, we fetch the image, turn it into a blob file,
    // and load it into the workspace so they can execute real ML predictions on it!
    const samples = {
        'Taj_Mahal': 'https://images.unsplash.com/photo-1564507592333-c60657eea523?auto=format&fit=crop&w=600&q=80',
        'Eiffel_Tower': 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=600&q=80',
        'Colosseum': 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=600&q=80',
        'Pyramids_of_Giza': 'https://images.unsplash.com/photo-1539650116574-8efeb43e2750?auto=format&fit=crop&w=600&q=80',
        'Statue_of_Liberty': 'https://images.unsplash.com/photo-1524008279394-3a9414322680?auto=format&fit=crop&w=600&q=80',
        'Machu_Picchu': 'https://images.unsplash.com/photo-1509024644558-2f56ce76c490?auto=format&fit=crop&w=600&q=80',
        'Great_Wall_of_China': 'https://images.unsplash.com/photo-1508807526345-15e9b7f430dd?auto=format&fit=crop&w=600&q=80',
        'Sydney_Opera_House': 'https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9?auto=format&fit=crop&w=600&q=80',
        'Stonehenge': 'https://images.unsplash.com/photo-1599833613876-4d2d48a3c530?auto=format&fit=crop&w=600&q=80',
        'Golden_Gate_Bridge': 'https://images.unsplash.com/photo-1506012787146-f92b2d7d6d96?auto=format&fit=crop&w=600&q=80'
    };
    
    const url = samples[landmarkKey];
    if (!url) return;
    
    // Switch to upload mode tab visual
    switchInputMode('upload');
    
    // Show spinner in dropzone during downloading
    const dropzone = document.getElementById('dropzone');
    const originalContent = dropzone.innerHTML;
    dropzone.innerHTML = `
        <div class="overlay-spinner" style="margin: 20px auto;"></div>
        <p class="primary-prompt">Fetching gallery image...</p>
    `;
    
    try {
        const response = await fetch(url);
        const blob = await response.blob();
        const file = new File([blob], `${landmarkKey}.jpg`, { type: 'image/jpeg' });
        
        dropzone.innerHTML = originalContent;
        setupDragAndDrop(); // Rebind events since HTML changed
        
        handleFileSelection(file);
    } catch (err) {
        console.error('Error loading sample image:', err);
        dropzone.innerHTML = originalContent;
        setupDragAndDrop();
        alert('Failed to load sample image. Please verify internet connection.');
    }
}

// ML Prediction Executor
// ML Prediction Executor
async function runPrediction() {
    if (!selectedFile || isProcessing) return;
    
    isProcessing = true;
    
    // Toggle active processing animations
    const overlay = document.getElementById('processing-overlay');
    const scanBar = document.getElementById('scan-bar');
    const btnPredict = document.getElementById('btn-predict');
    
    // BATCH MODE PIPELINE
    if (Array.isArray(selectedFile)) {
        const files = selectedFile;
        btnPredict.disabled = true;
        
        let successCount = 0;
        let lastResult = null;
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const itemEl = document.getElementById(`batch-item-${i}`);
            
            // Set status to processing
            if (itemEl) {
                itemEl.className = 'batch-preview-item status-processing';
                itemEl.querySelector('.batch-status-badge').textContent = 'Analyzing...';
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error('Prediction API failed');
                
                const results = await response.json();
                lastResult = results;
                successCount++;
                
                // Update badge to done
                if (itemEl) {
                    itemEl.className = 'batch-preview-item status-done';
                    itemEl.querySelector('.batch-status-badge').textContent = 'Done';
                }
                
                // Mount prediction result locally and slide map
                predictionResults = results;
                renderPredictionOutput();
                
                let lat = null;
                let lng = null;
                if (results.is_landmark_detected && results.details) {
                    lat = results.details.latitude;
                    lng = results.details.longitude;
                }
                
                // Reload history and markers (pass active lat/lng coordinates to keep popup open)
                await loadHistory(lat, lng);
                
            } catch (err) {
                console.error(err);
                if (itemEl) {
                    itemEl.className = 'batch-preview-item status-error';
                    itemEl.querySelector('.batch-status-badge').textContent = 'Failed';
                }
            }
            
            // Pause 2 seconds to make sure the flyTo animation completes and results are visible
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
        
        btnPredict.disabled = false;
        isProcessing = false;
        
        // Notify user about progress status
        alert(`Batch Analysis Complete: successfully processed ${successCount}/${files.length} images.`);
        return;
    }
    
    // SINGLE FILE MODE PIPELINE
    overlay.classList.add('active');
    scanBar.classList.add('animating');
    btnPredict.disabled = true;
    
    const formData = new FormData();
    let options = {};
    
    if (typeof selectedFile === 'string') {
        // Webcam base64 string
        options = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: selectedFile })
        };
    } else {
        // Normal file upload
        formData.append('file', selectedFile);
        options = {
            method: 'POST',
            body: formData
        };
    }
    
    try {
        const response = await fetch('/predict', options);
        if (!response.ok) throw new Error('Prediction API failed');
        
        predictionResults = await response.json();
        
        // Hide loader & show result panels
        overlay.classList.remove('active');
        scanBar.classList.remove('animating');
        btnPredict.disabled = false;
        
        renderPredictionOutput();
        
        let lat = null;
        let lng = null;
        if (predictionResults.is_landmark_detected && predictionResults.details) {
            lat = predictionResults.details.latitude;
            lng = predictionResults.details.longitude;
        }
        loadHistory(lat, lng); // Reload history grid and open new popup
        isProcessing = false;
    } catch (err) {
        console.error(err);
        alert('Error compiling image predictions.');
        overlay.classList.remove('active');
        scanBar.classList.remove('animating');
        btnPredict.disabled = false;
        isProcessing = false;
    }
}

// Display results on UI
function renderPredictionOutput() {
    const res = predictionResults;
    if (!res) return;
    
    // Hide empty state and show results panel
    document.getElementById('results-empty-state').style.display = 'none';
    document.getElementById('results-active-panel').style.display = 'flex';
    
    // Force Leaflet to recalculate dimensions after display changes to flex
    setTimeout(() => {
        if (mapInstance) {
            mapInstance.invalidateSize();
        }
    }, 100);
    
    // 1. Header Details
    const nameEl = document.getElementById('res-landmark-name');
    const confidenceEl = document.getElementById('res-confidence');
    const cityEl = document.getElementById('res-city');
    const countryEl = document.getElementById('res-country');
    const weatherEl = document.getElementById('res-weather-pill');
    
    nameEl.textContent = res.landmark_name;
    confidenceEl.textContent = `${Math.round(res.confidence * 100)}%`;
    
    if (res.is_landmark_detected && res.details) {
        cityEl.style.display = 'inline-flex';
        countryEl.style.display = 'inline-flex';
        cityEl.innerHTML = `<i class="fa-solid fa-city"></i> ${res.details.city || 'Unknown'}`;
        countryEl.innerHTML = `<i class="fa-solid fa-location-dot"></i> ${res.details.country || 'Unknown'}`;
        
        // Show live weather conditions
        if (res.weather) {
            weatherEl.style.display = 'inline-flex';
            weatherEl.innerHTML = `<i class="fa-solid ${res.weather.icon}"></i> ${res.weather.temp} (${res.weather.condition})`;
        } else {
            weatherEl.style.display = 'none';
        }
    } else {
        cityEl.style.display = 'none';
        countryEl.style.display = 'none';
        weatherEl.style.display = 'none';
    }
    
    // 2. Images (Original & Heatmap)
    document.getElementById('res-original-image').src = res.image_url;
    const heatmapImage = document.getElementById('res-heatmap-image');
    const heatmapBtn = document.getElementById('btn-show-heatmap');
    
    if (res.heatmap_url) {
        heatmapImage.src = res.heatmap_url;
        heatmapBtn.style.display = 'inline-block';
    } else {
        heatmapBtn.style.display = 'none';
    }
    
    // Reset visual mode toggle to original image
    toggleVisualMode('original');
    
    // 3. Render Top 5 Predictions List
    const predictionsList = document.getElementById('res-predictions-list');
    predictionsList.innerHTML = '';
    
    res.top_5.forEach(pred => {
        const row = document.createElement('div');
        row.className = 'prediction-bar-row';
        
        const pct = `${Math.round(pred.confidence * 100)}%`;
        row.innerHTML = `
            <span class="bar-label" title="${pred.readable_name}">${pred.readable_name}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width: 0%;"></div>
            </div>
            <span class="bar-value">${pct}</span>
        `;
        predictionsList.appendChild(row);
        
        // Trigger animations width after rendering
        setTimeout(() => {
            row.querySelector('.bar-fill').style.width = pct;
        }, 100);
    });
    
    // 4. Render Landmark Information Panel
    const metadataCard = document.getElementById('landmark-metadata-card');
    
    if (res.is_landmark_detected && res.details) {
        metadataCard.style.display = 'block';
        
        // Details
        document.getElementById('md-history-text').textContent = res.details.historical_background || 'No historical data available.';
        
        // Key Facts
        const factsList = document.getElementById('md-facts-list');
        factsList.innerHTML = '';
        if (res.details.interesting_facts && Array.isArray(res.details.interesting_facts)) {
            res.details.interesting_facts.forEach(fact => {
                const li = document.createElement('li');
                li.textContent = fact;
                factsList.appendChild(li);
            });
        }
        
        // Architectural Details
        document.getElementById('spec-built').textContent = res.details.built_year || 'N/A';
        document.getElementById('spec-architect').textContent = res.details.architect || 'N/A';
        document.getElementById('spec-style').textContent = res.details.architectural_style || 'N/A';
        document.getElementById('spec-unesco').textContent = res.details.unesco_status || 'N/A';
        
        // Tourist Details
        document.getElementById('tr-hours').textContent = res.details.opening_hours || 'N/A';
        document.getElementById('tr-fee').textContent = res.details.entry_fee || 'N/A';
        document.getElementById('tr-best-time').textContent = res.details.best_time_to_visit || 'N/A';
        document.getElementById('tr-tips').textContent = res.details.tourist_tips || 'N/A';
        
        // Force reset to first tab
        switchMetaTab('history');
        
        // 5. Focus Leaflet Interactive Map & 360° Virtual Exploration
        if (res.details.latitude !== undefined && res.details.longitude !== undefined) {
            const lat = res.details.latitude;
            const lng = res.details.longitude;
            
            document.getElementById('res-coordinates').textContent = `LAT: ${lat.toFixed(4)}, LNG: ${lng.toFixed(4)}`;
            
            // Set 360° Google Maps Street View Embeds
            const embedUrl = `https://maps.google.com/maps?q=${lat},${lng}&layer=c&cbll=${lat},${lng}&cbp=12,0,0,0,0&output=embed`;
            
            const tabIframe = document.getElementById('tab-streetview-iframe');
            if (tabIframe) tabIframe.src = embedUrl;
            
            const mapIframe = document.getElementById('map-streetview-iframe');
            if (mapIframe) mapIframe.src = embedUrl;
            
            // Set External Google Earth & Street View Links
            const btnEarth = document.getElementById('btn-google-earth');
            if (btnEarth) {
                btnEarth.href = `https://earth.google.com/web/search/${encodeURIComponent(res.landmark_name)}`;
            }
            
            const btnStreetView = document.getElementById('btn-google-streetview-external');
            if (btnStreetView) {
                btnStreetView.href = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat},${lng}`;
            }

            if (!mapInstance) {
                initMap();
            }
            mapInstance.flyTo([lat, lng], 13, {
                animate: true,
                duration: 1.5
            });
            
            // Trigger weather overlay animation based on local conditions
            updateMeteorologicalOverlay(res.weather);
        } else {
            document.getElementById('res-coordinates').textContent = 'Coordinates Unavailable';
            clearWeatherEffects();
        }
    } else {
        metadataCard.style.display = 'none';
        document.getElementById('res-coordinates').textContent = 'Location Unknown';
    }
}

// Map management logic
// Initialize Map Once
function initMap() {
    const mapContainer = document.getElementById('landmark-map');
    if (!mapContainer || mapInstance) return;
    
    // Set style default
    currentMapStyle = 'street';
    const streetBtn = document.getElementById('btn-map-street');
    const satBtn = document.getElementById('btn-map-satellite');
    if (streetBtn) streetBtn.classList.add('active');
    if (satBtn) satBtn.classList.remove('active');
    
    // Init Leaflet instance showing whole world map
    mapInstance = L.map('landmark-map', { zoomControl: false }).setView([20, 0], 2);
    
    // Add Zoom Control
    L.control.zoom({ position: 'topleft' }).addTo(mapInstance);
    
    // Store Tile Layers
    mapTileLayers.street = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    });
    
    mapTileLayers.satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    });
    
    // Default load street layer
    mapTileLayers.street.addTo(mapInstance);
}

// Plot Markers for all History Items
function plotHistoryMarkers(history, activeLat = null, activeLng = null) {
    if (!mapInstance) return;
    
    // Recalculate container size to fix Leaflet gray container bug
    mapInstance.invalidateSize();
    
    // Clear all existing markers from map
    mapMarkers.forEach(m => mapInstance.removeLayer(m));
    mapMarkers = [];
    if (mapMarker) {
        mapInstance.removeLayer(mapMarker);
        mapMarker = null;
    }
    
    // Clear existing polyline route
    if (travelRouteLine) {
        mapInstance.removeLayer(travelRouteLine);
        travelRouteLine = null;
    }
    
    let activeMarker = null;
    let routeCoords = [];
    
    // Custom neon pulsing Leaflet divIcon
    const neonIcon = L.divIcon({
        className: 'custom-neon-marker-container',
        html: `
            <div class="custom-neon-marker">
                <div class="neon-pulse-ring"></div>
                <div class="neon-dot"></div>
            </div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });
    
    // Loop through history and place pins
    // History is sorted newest-first. Let's iterate in reverse order (oldest to newest) to draw the route correctly!
    const sortedHistory = [...history].reverse();
    
    sortedHistory.forEach(item => {
        if (item.latitude !== undefined && item.latitude !== null &&
            item.longitude !== undefined && item.longitude !== null) {
            
            const confidencePct = `${Math.round(item.confidence * 100)}%`;
            
            // Custom HTML popup with image thumbnail preview!
            const popupContent = `
                <div class="map-popup-card" style="min-width: 140px;">
                    <div class="map-popup-img">
                        <img src="${item.image_url}" alt="${item.landmark_name}" style="width: 100%; max-height: 85px; object-fit: cover; border-radius: 4px; margin-bottom: 4px;">
                    </div>
                    <div class="map-popup-info">
                        <b style="color: #a855f7; font-size: 13px; font-family: 'Outfit', sans-serif;">${item.landmark_name}</b><br>
                        <span style="font-size: 11px; color: #cbd5e1;">Confidence: <b>${item.is_detected ? confidencePct : 'Unknown'}</b></span><br>
                        <span style="font-size: 9px; color: #94a3b8;">${item.timestamp}</span>
                    </div>
                </div>
            `;
            
            const marker = L.marker([item.latitude, item.longitude], { icon: neonIcon })
                .addTo(mapInstance)
                .bindPopup(popupContent);
                
            mapMarkers.push(marker);
            
            // Add coordinates for route line
            routeCoords.push([item.latitude, item.longitude]);
            
            // Reference match for newly predicted marker popup display
            if (activeLat !== null && activeLng !== null &&
                Math.abs(item.latitude - activeLat) < 0.0001 &&
                Math.abs(item.longitude - activeLng) < 0.0001) {
                activeMarker = marker;
            }
        }
    });
    
    // Draw the travel route polyline connecting the locations chronologically
    if (routeCoords.length > 1) {
        travelRouteLine = L.polyline(routeCoords, {
            color: '#a855f7', // neon purple
            weight: 3,
            opacity: 0.7,
            dashArray: '6, 8', // dashed pattern
            lineJoin: 'round',
            className: 'neon-route-path' // for animated flowing CSS dash effect
        }).addTo(mapInstance);
    }
    
    if (activeMarker) {
        activeMarker.openPopup();
    }
}

function switchMapStyle(style) {
    if (!mapInstance) return;
    
    const streetBtn = document.getElementById('btn-map-street');
    const satBtn = document.getElementById('btn-map-satellite');
    const panoBtn = document.getElementById('btn-map-panorama');
    const mapDiv = document.getElementById('landmark-map');
    const panoDiv = document.getElementById('map-streetview-container');
    
    currentMapStyle = style;
    
    if (style === 'panorama') {
        if (mapDiv) mapDiv.style.display = 'none';
        if (panoDiv) panoDiv.style.display = 'block';
        
        if (streetBtn) streetBtn.classList.remove('active');
        if (satBtn) satBtn.classList.remove('active');
        if (panoBtn) panoBtn.classList.add('active');
    } else {
        if (mapDiv) mapDiv.style.display = 'block';
        if (panoDiv) panoDiv.style.display = 'none';
        
        if (mapTileLayers.street) mapInstance.removeLayer(mapTileLayers.street);
        if (mapTileLayers.satellite) mapInstance.removeLayer(mapTileLayers.satellite);
        if (mapTileLayers[style]) mapTileLayers[style].addTo(mapInstance);
        
        if (streetBtn) streetBtn.classList.toggle('active', style === 'street');
        if (satBtn) satBtn.classList.toggle('active', style === 'satellite');
        if (panoBtn) panoBtn.classList.remove('active');
        
        setTimeout(() => mapInstance.invalidateSize(), 50);
    }
}

// Switch Visual View modes: Original image vs Attention Heatmap
function toggleVisualMode(mode) {
    if (activeVisualMode === mode) return;
    
    activeVisualMode = mode;
    
    const originalImage = document.getElementById('res-original-image');
    const heatmapImage = document.getElementById('res-heatmap-image');
    const labelMode = document.getElementById('visual-label-mode');
    
    if (mode === 'original') {
        originalImage.style.display = 'block';
        heatmapImage.style.display = 'none';
        labelMode.textContent = 'Original Input';
    } else {
        originalImage.style.display = 'none';
        heatmapImage.style.display = 'block';
        labelMode.textContent = 'Grad-CAM Focus Area';
    }
    
    document.getElementById('btn-show-original').classList.toggle('active', mode === 'original');
    document.getElementById('btn-show-heatmap').classList.toggle('active', mode === 'heatmap');
}

// Details Metadata Section Tab Toggles
function switchMetaTab(tab) {
    if (activeMetaTab === tab) return;
    
    activeMetaTab = tab;
    
    // Select and toggle tab content containers
    const tabHistory = document.getElementById('meta-history');
    const tabArchitecture = document.getElementById('meta-architecture');
    const tabTourist = document.getElementById('meta-tourist');
    const tabVirtual360 = document.getElementById('meta-virtual360');
    
    if (tabHistory) tabHistory.classList.toggle('active', tab === 'history');
    if (tabArchitecture) tabArchitecture.classList.toggle('active', tab === 'architecture');
    if (tabTourist) tabTourist.classList.toggle('active', tab === 'tourist');
    if (tabVirtual360) tabVirtual360.classList.toggle('active', tab === 'virtual360');
    
    // Select and toggle buttons
    const btnHistory = document.getElementById('tab-history');
    const btnArchitecture = document.getElementById('tab-architecture');
    const btnTourist = document.getElementById('tab-tourist');
    const btnVirtual360 = document.getElementById('tab-virtual360');
    
    if (btnHistory) btnHistory.classList.toggle('active', tab === 'history');
    if (btnArchitecture) btnArchitecture.classList.toggle('active', tab === 'architecture');
    if (btnTourist) btnTourist.classList.toggle('active', tab === 'tourist');
    if (btnVirtual360) btnVirtual360.classList.toggle('active', tab === 'virtual360');
}

// Meteorological Map Weather Overlay Effects
function clearWeatherEffects() {
    const container = document.getElementById('map-weather-overlay');
    if (container) {
        container.innerHTML = '';
        container.className = 'map-weather-overlay-layer';
    }
}

function startRainEffect() {
    const container = document.getElementById('map-weather-overlay');
    if (!container) return;
    container.innerHTML = '';
    container.className = 'map-weather-overlay-layer rain-active';
    
    const dropCount = 40;
    for (let i = 0; i < dropCount; i++) {
        const drop = document.createElement('div');
        drop.className = 'weather-rain-drop';
        drop.style.left = `${Math.random() * 100}%`;
        drop.style.animationDelay = `${Math.random() * 2}s`;
        drop.style.animationDuration = `${0.5 + Math.random() * 0.5}s`;
        drop.style.opacity = `${0.2 + Math.random() * 0.4}`;
        container.appendChild(drop);
    }
}

function startSnowEffect() {
    const container = document.getElementById('map-weather-overlay');
    if (!container) return;
    container.innerHTML = '';
    container.className = 'map-weather-overlay-layer snow-active';
    
    const flakeCount = 25;
    for (let i = 0; i < flakeCount; i++) {
        const flake = document.createElement('div');
        flake.className = 'weather-snow-flake';
        flake.style.left = `${Math.random() * 100}%`;
        flake.style.animationDelay = `${Math.random() * 5}s`;
        flake.style.animationDuration = `${3 + Math.random() * 3}s`;
        flake.style.opacity = `${0.3 + Math.random() * 0.5}`;
        flake.style.width = flake.style.height = `${2 + Math.random() * 4}px`;
        container.appendChild(flake);
    }
}

function startThunderEffect() {
    startRainEffect();
    const container = document.getElementById('map-weather-overlay');
    if (container) {
        container.classList.add('thunder-active');
    }
}

function updateMeteorologicalOverlay(weather) {
    clearWeatherEffects();
    if (!weather || !weather.condition) return;
    
    const condition = weather.condition.toLowerCase();
    if (condition.includes('rain') || condition.includes('drizzle') || condition.includes('showers')) {
        if (condition.includes('thunderstorm') || condition.includes('storm')) {
            startThunderEffect();
        } else {
            startRainEffect();
        }
    } else if (condition.includes('snow') || condition.includes('freezing')) {
        startSnowEffect();
    }
}

// 3D Telemetry Background Globe using Three.js
let globeMesh, globeParticles, globeRenderer, globeScene, globeCamera;

function initBackgroundGlobe() {
    const canvas = document.getElementById('globe-bg-canvas');
    if (!canvas) return;
    
    // Check if Three.js is loaded
    if (typeof THREE === 'undefined') {
        console.warn('Three.js library not loaded. Telemetry globe background skipped.');
        return;
    }
    
    try {
        // Create Scene, Camera, Renderer
        globeScene = new THREE.Scene();
        globeCamera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        globeCamera.position.z = 15;
        
        globeRenderer = new THREE.WebGLRenderer({
            canvas: canvas,
            alpha: true,
            antialias: true
        });
        globeRenderer.setPixelRatio(window.devicePixelRatio);
        globeRenderer.setSize(window.innerWidth, window.innerHeight);
        
        // Group to hold globe components
        const globeGroup = new THREE.Group();
        globeScene.add(globeGroup);
        
        // 1. Core Wireframe Sphere (Earth grid structure)
        const globeGeometry = new THREE.SphereGeometry(6.5, 30, 25);
                const globeMaterial = new THREE.MeshBasicMaterial({
            color: 0xa855f7, // Neon Purple
            wireframe: true,
            transparent: true,
            opacity: 0.22
        });
        globeMesh = new THREE.Mesh(globeGeometry, globeMaterial);
        globeGroup.add(globeMesh);
        
        // 2. Secondary Particle Cloud (Dotted matrix outline)
        const particleGeometry = new THREE.SphereGeometry(6.7, 45, 35);
        const particleCount = particleGeometry.attributes.position.count;
        const particlePos = particleGeometry.attributes.position;
        
        const dotsGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = particlePos.getX(i);
            positions[i * 3 + 1] = particlePos.getY(i);
            positions[i * 3 + 2] = particlePos.getZ(i);
        }
        
        dotsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        // Custom canvas circle dot texture for high quality dot rendering
        const dotCanvas = document.createElement('canvas');
        dotCanvas.width = 16;
        dotCanvas.height = 16;
        const ctx = dotCanvas.getContext('2d');
        const grad = ctx.createRadialGradient(8, 8, 0, 8, 8, 8);
        grad.addColorStop(0, 'rgba(6, 182, 212, 0.95)'); // Cyber Cyan
        grad.addColorStop(1, 'rgba(6, 182, 212, 0)');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 16, 16);
        
        const dotTexture = new THREE.CanvasTexture(dotCanvas);
        
                const particleMaterial = new THREE.PointsMaterial({
            size: 0.15,
            map: dotTexture,
            transparent: true,
            opacity: 0.45,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        
        globeParticles = new THREE.Points(dotsGeometry, particleMaterial);
        globeGroup.add(globeParticles);
        
        // Render loop
        function draw() {
            if (!globeRenderer) return;
            requestAnimationFrame(draw);
            
            // Slow continuous rotation
            globeGroup.rotation.y += 0.0008;
            globeGroup.rotation.x += 0.0003;
            
            // Subtle hover ripple
            globeParticles.rotation.y -= 0.0002;
            
            globeRenderer.render(globeScene, globeCamera);
        }
        
        draw();
        
        // Window Resize synchronization
        window.addEventListener('resize', onGlobeWindowResize);
    } catch (e) {
        console.error('Failed to initialize WebGL background globe:', e);
    }
}

function onGlobeWindowResize() {
    if (!globeCamera || !globeRenderer) return;
    globeCamera.aspect = window.innerWidth / window.innerHeight;
    globeCamera.updateProjectionMatrix();
    globeRenderer.setSize(window.innerWidth, window.innerHeight);
}
