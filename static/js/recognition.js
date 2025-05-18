// static/js/recognition.js

document.addEventListener('DOMContentLoaded', () => {
    // --- Globals ---
    const socket = io(); // Connect to Socket.IO server
    let videoStream = null;
    let mediaRecorder = null; // Not used for frame-by-frame, but good to have if switching later
    let collectedLiveLandmarks = []; // For webcam frames sent one-by-one
    let countdownInterval = null;
    let frameProcessingInterval = null;
    let isCapturing = false;
    const frameRate = 15; // FPS for webcam capture

    // --- DOM Elements ---
    const webcamFeed = document.getElementById('webcamFeed');
    const canvas = document.getElementById('canvas'); // Hidden canvas
    const ctx = canvas.getContext('2d');

    const enableCamButton = document.getElementById('enableCamButton');
    const startCaptureButton = document.getElementById('startCaptureButton');
    const stopCaptureButton = document.getElementById('stopCaptureButton');
    const recIndicatorWebcam = document.getElementById('recIndicatorWebcam');
    const countdownDisplay = document.getElementById('countdownDisplay');

    const uploadForm = document.getElementById('uploadForm');
    const videoFile = document.getElementById('videoFile');
    const uploadVideoButton = document.getElementById('uploadVideoButton'); // Submit button for form

    const statusMessage = document.getElementById('statusMessage');
    const resultsArea = document.getElementById('resultsArea');
    const predictionList = document.getElementById('predictionList');
    const confidenceMessage = document.getElementById('confidenceMessage');
    const playbackLink = document.getElementById('playbackLink');
    const downloadRawJsonLink = document.getElementById('downloadRawJsonLink'); // New link element
    const playbackLinkContainer = document.getElementById('playbackLinkContainer'); // To show/hide link
    const clearResultsButton = document.getElementById('clearResultsButton');

    // Model selector elements
    const modelSelector = document.getElementById('modelSelector');
    const modelStatus = document.getElementById('modelStatus');

    // --- Initial UI State ---
    function resetUIForNewSign() {
        stopCaptureButton.disabled = true;
        startCaptureButton.disabled = !videoStream; // Enable if cam is already on
        enableCamButton.disabled = !!videoStream;  // Disable if cam is on
        
        uploadForm.reset(); // Clear file input
        uploadVideoButton.disabled = false;
        videoFile.disabled = false;

        resultsArea.classList.add('hidden');
        predictionList.innerHTML = '';
        confidenceMessage.textContent = '';
        playbackLink.classList.add('hidden');
        playbackLink.href = '#';
        if (downloadRawJsonLink) { // Ensure it's hidden and reset
            downloadRawJsonLink.classList.add('hidden');
            downloadRawJsonLink.href = '#';
            downloadRawJsonLink.removeAttribute('download');
        }
        countdownDisplay.textContent = '';
        recIndicatorWebcam.style.visibility = 'hidden';
        statusMessage.textContent = videoStream ? 'Ready to capture or upload.' : 'Please enable camera or upload a video.';
        isCapturing = false;
        collectedLiveLandmarks = [];
    }
    
    resetUIForNewSign(); // Set initial state

    // --- Model Selector Logic ---
    async function loadAvailableModels() {
        try {
            modelStatus.textContent = 'Loading...';
            modelStatus.className = 'status-loading';
            
            const response = await fetch('/models');
            const data = await response.json();
            
            // Clear existing options
            modelSelector.innerHTML = '';
            
            // Add options for each available model
            data.available_models.forEach(modelName => {
                const option = document.createElement('option');
                option.value = modelName;
                option.textContent = modelName.charAt(0).toUpperCase() + modelName.slice(1); // Capitalize first letter
                option.selected = modelName === data.current_model;
                modelSelector.appendChild(option);
            });
            
            modelStatus.textContent = `Using: ${data.current_model}`;
            modelStatus.className = 'status-success';
        } catch (error) {
            console.error('Error loading models:', error);
            modelStatus.textContent = 'Error loading models';
            modelStatus.className = 'status-error';
        }
    }
    
    // Load models when page loads
    loadAvailableModels();
    
    // Handle model change
    modelSelector.addEventListener('change', async function() {
        const selectedModel = this.value;
        
        try {
            modelStatus.textContent = 'Changing...';
            modelStatus.className = 'status-loading';
            
            const response = await fetch('/change_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_name: selectedModel }),
            });
            
            const result = await response.json();
            
            if (response.ok) {
                modelStatus.textContent = `Using: ${result.current_model}`;
                modelStatus.className = 'status-success';
                
                // Update status message
                updateStatus(`Model changed to: ${result.current_model}`);
            } else {
                modelStatus.textContent = result.error || 'Error changing model';
                modelStatus.className = 'status-error';
                
                // Reset the selector to the current model
                loadAvailableModels();
            }
        } catch (error) {
            console.error('Error changing model:', error);
            modelStatus.textContent = 'Error changing model';
            modelStatus.className = 'status-error';
            
            // Reset the selector to the current model
            loadAvailableModels();
        }
    });

    // --- Helper Functions ---
    function updateStatus(message, isError = false) {
        statusMessage.textContent = message;
        statusMessage.style.color = isError ? 'red' : 'black';
    }

    function displayPredictions(predictionsData) {
        predictionList.innerHTML = ''; // Clear previous predictions
        if (!predictionsData || predictionsData.length === 0) {
            predictionList.innerHTML = '<li>No predictions returned.</li>';
            resultsArea.classList.remove('hidden');
            return;
        }

        let overallConfidenceLow = true;
        predictionsData.forEach(pred => {
            const listItem = document.createElement('li');
            listItem.innerHTML = `<span class="label">${pred.label}:</span> <span class="confidence">${pred.confidence.toFixed(2)}%</span>`;
            predictionList.appendChild(listItem);
            if (pred.confidence > 20) { // Arbitrary threshold for "not low"
                overallConfidenceLow = false;
            }
        });

        if (overallConfidenceLow && predictionsData.length > 0 && predictionsData[0].confidence < 10) { // If top prediction is very low
             confidenceMessage.textContent = "I'm not very sure about these. Try signing more clearly, check lighting, or ensure the sign is in the vocabulary.";
        } else if (predictionsData[0] && predictionsData[0].label === "No significant motion detected") {
             confidenceMessage.textContent = "It seems like no significant motion was detected in the video.";
        }
        else {
            confidenceMessage.textContent = ''; // Clear if confident enough
        }
        resultsArea.classList.remove('hidden');
    }


    // --- Webcam Logic ---
    enableCamButton.addEventListener('click', async () => {
        if (videoStream) return; // Already enabled
        updateStatus('Initializing webcam...');
        try {
            videoStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            webcamFeed.srcObject = videoStream;
            await new Promise(resolve => webcamFeed.onloadedmetadata = resolve); // Wait for metadata
            
            // Set canvas dimensions once video is loaded
            canvas.width = webcamFeed.videoWidth;
            canvas.height = webcamFeed.videoHeight;

            updateStatus('Webcam enabled. Ready to capture.');
            enableCamButton.disabled = true;
            startCaptureButton.disabled = false;
            stopCaptureButton.disabled = true;
        } catch (err) {
            console.error("Error accessing webcam:", err);
            updateStatus(`Error enabling webcam: ${err.message}`, true);
            videoStream = null;
        }
    });

    startCaptureButton.addEventListener('click', () => {
        if (!videoStream) {
            updateStatus('Please enable the camera first.', true);
            return;
        }
        isCapturing = true;
        collectedLiveLandmarks = []; // Reset for new capture

        startCaptureButton.disabled = true;
        stopCaptureButton.disabled = false;
        uploadForm.reset(); // Disable upload while capturing
        uploadVideoButton.disabled = true;
        videoFile.disabled = true;
        resultsArea.classList.add('hidden'); // Hide old results

        let count = 3;
        countdownDisplay.textContent = count;
        updateStatus('Get ready...');

        countdownInterval = setInterval(() => {
            count--;
            countdownDisplay.textContent = count > 0 ? count : 'GO!';
            if (count <= 0) {
                clearInterval(countdownInterval);
                countdownDisplay.textContent = '';
                updateStatus('Capturing...');
                recIndicatorWebcam.style.visibility = 'visible';
                
                // Start sending frames
                frameProcessingInterval = setInterval(() => {
                    if (!isCapturing) {
                        clearInterval(frameProcessingInterval);
                        recIndicatorWebcam.style.visibility = 'hidden';
                        return;
                    }
                    if (webcamFeed.readyState >= webcamFeed.HAVE_CURRENT_DATA && webcamFeed.videoWidth > 0) {
                        if (canvas.width !== webcamFeed.videoWidth || canvas.height !== webcamFeed.videoHeight) {
                             canvas.width = webcamFeed.videoWidth;
                             canvas.height = webcamFeed.videoHeight;
                        }
                        ctx.drawImage(webcamFeed, 0, 0, canvas.width, canvas.height);
                        const frameData = canvas.toDataURL('image/jpeg', 0.8); // Quality 0.8
                        socket.emit('live_frame_for_recognition', frameData);
                    }
                }, 1000 / frameRate);
            }
        }, 1000);
    });

    stopCaptureButton.addEventListener('click', () => {
        isCapturing = false; // This will stop the interval in its next check
        if(frameProcessingInterval) clearInterval(frameProcessingInterval);
        if(countdownInterval) clearInterval(countdownInterval); // Ensure countdown stops if stop is clicked early

        recIndicatorWebcam.style.visibility = 'hidden';
        stopCaptureButton.disabled = true;
        startCaptureButton.disabled = false; // Can start a new capture
        uploadVideoButton.disabled = false; // Re-enable upload
        videoFile.disabled = false;


        if (collectedLiveLandmarks.length > 0) {
            updateStatus(`Processing ${collectedLiveLandmarks.length} captured frames...`);
            // Send the whole sequence for prediction
            socket.emit('predict_webcam_sequence', collectedLiveLandmarks);
        } else {
            updateStatus('No frames captured or landmarks extracted. Please try again.', true);
            resetUIForNewSign(); // Or just enable start button
        }
    });

    // --- Video Upload Logic ---
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!videoFile.files || videoFile.files.length === 0) {
            updateStatus('Please select a video file to upload.', true);
            return;
        }
        updateStatus('Uploading and processing video...');
        uploadVideoButton.disabled = true;
        startCaptureButton.disabled = true; // Disable webcam capture during upload processing
        enableCamButton.disabled = true;

        const formData = new FormData();
        formData.append('video', videoFile.files[0]);

        try {
            const response = await fetch('/upload_and_predict', {
                method: 'POST',
                body: formData,
            });
            const result = await response.json();

            if (response.ok) {
                updateStatus('Video processed successfully!');
                displayPredictions(result.predictions);
                if (result.playback_file) {
                    const fileUrlForPlayback = `/playback?file_url=/data/${result.playback_file}`;
                    const fileUrlForDownload = `/data/${result.playback_file}`;

                    playbackLink.href = fileUrlForPlayback;
                    playbackLink.classList.remove('hidden');

                    if (downloadRawJsonLink) {
                        downloadRawJsonLink.href = fileUrlForDownload;
                        downloadRawJsonLink.download = result.playback_file; // Set the filename for download
                        downloadRawJsonLink.classList.remove('hidden');
                    }
                    playbackLinkContainer.classList.remove('hidden'); // Ensure container is visible
                } else { // If no file, ensure both links are hidden
                    playbackLink.classList.add('hidden');
                    if (downloadRawJsonLink) downloadRawJsonLink.classList.add('hidden');
                }
            } else {
                updateStatus(`Error: ${result.error || 'Upload failed'}`, true);
            }
        } catch (error) {
            console.error('Upload error:', error);
            updateStatus(`Error: ${error.message}`, true);
        } finally {
            uploadVideoButton.disabled = false;
            // Re-enable webcam based on its state
            if(videoStream) {
                startCaptureButton.disabled = false;
                enableCamButton.disabled = true;
            } else {
                startCaptureButton.disabled = true;
                enableCamButton.disabled = false;
            }
            videoFile.value = ''; // Clear file input
        }
    });
    
    // --- Clear/Reset Button ---
    clearResultsButton.addEventListener('click', resetUIForNewSign);


    // --- SocketIO Event Listeners ---
    socket.on('connect', () => {
        console.log('Connected to server (recognition.js)');
        updateStatus(videoStream ? 'Webcam ready. Or upload a video.' : 'Please enable camera or upload a video.');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateStatus('Disconnected. Please refresh.', true);
        // Potentially stop ongoing capture if any
        if (isCapturing) {
            stopCaptureButton.click(); // Simulate click to stop and clean up
        }
        startCaptureButton.disabled = true;
        stopCaptureButton.disabled = true;
        enableCamButton.disabled = false; // Allow re-enabling
    });

    socket.on('live_landmarks_result', (data) => {
        if (isCapturing) { // Only collect if we are actively capturing
            if (data.error) {
                 console.error("Backend Error (live_landmarks_result):", data.error);
                 // Optionally display this error to the user, but be mindful of spamming
            } else if (data.landmarks && data.landmarks.length > 0) {
                collectedLiveLandmarks.push(data.landmarks);
                // Optionally update a live frame count, but status message might be enough
                // document.getElementById('liveFrameCount').textContent = collectedLiveLandmarks.length;
            } else if (data.landmarks && data.landmarks.length === 0) {
                // Server might send empty landmarks if nothing detected in frame, still collect it as a frame
                collectedLiveLandmarks.push([]); // Or push a specific marker for "empty detection"
            }
        }
    });

    socket.on('webcam_prediction_result', (result) => {
        if (result.error) {
            updateStatus(`Prediction error: ${result.error}`, true);
        } else {
            updateStatus('Webcam capture processed.');
            displayPredictions(result.predictions);
            if (result.playback_file) {
                const fileUrlForPlayback = `/playback?file_url=/data/${result.playback_file}`;
                const fileUrlForDownload = `/data/${result.playback_file}`;

                playbackLink.href = fileUrlForPlayback;
                playbackLink.classList.remove('hidden');

                if (downloadRawJsonLink) {
                    downloadRawJsonLink.href = fileUrlForDownload;
                    downloadRawJsonLink.download = result.playback_file; // Set the filename for download
                    downloadRawJsonLink.classList.remove('hidden');
                }
                playbackLinkContainer.classList.remove('hidden'); // Make sure container is visible
            } else { // If no file, ensure both links are hidden
                playbackLink.classList.add('hidden');
                if (downloadRawJsonLink) downloadRawJsonLink.classList.add('hidden');
            }
        }
        // UI should be mostly reset by stopCaptureButton logic already
    });

}); // End DOMContentLoaded