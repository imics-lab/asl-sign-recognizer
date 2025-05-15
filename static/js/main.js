// static/js/main.js
// This script is now specifically for landmark_extractor.html

document.addEventListener('DOMContentLoaded', () => {
    // --- Globals ---
    const socket = io(); // Connect to Socket.IO server
    let videoStreamExtractor = null;
    let processingIntervalExtractor = null;
    let collectedLandmarksExtractor = [];
    const frameRateExtractor = 15; // FPS

    // --- DOM Elements (ensure these IDs match landmark_extractor.html) ---
    const uploadForm = document.getElementById('uploadFormExtractor');
    const videoFile = document.getElementById('videoFileExtractor');
    const uploadStatus = document.getElementById('uploadStatusExtractor');
    const uploadResultsArea = document.getElementById('uploadResultsAreaExtractor');
    const uploadResultData = document.getElementById('uploadResultDataExtractor');
    const downloadLink = document.getElementById('downloadLinkExtractor');

    const webcamFeed = document.getElementById('webcamFeedExtractor');
    const canvas = document.getElementById('canvasExtractor'); 
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('startButtonExtractor');
    const stopButton = document.getElementById('stopButtonExtractor');
    const status = document.getElementById('statusExtractor'); // Corrected ID
    const resultsArea = document.getElementById('resultsAreaExtractor'); // Corrected ID
    const frameCount = document.getElementById('frameCountExtractor'); // Corrected ID
    const webcamDownloadLink = document.getElementById('webcamDownloadLinkExtractor'); // Corrected ID

    // --- File Upload Logic ---
    if (uploadForm) { // Check if the element exists to prevent errors if not on the page
        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (!videoFile.files || videoFile.files.length === 0) {
                uploadStatus.textContent = 'Please select a video file.';
                return;
            }
            uploadStatus.textContent = 'Uploading and processing for extraction...';
            uploadResultsArea.classList.add('hidden');
            if(downloadLink) downloadLink.style.display = 'none';

            const formData = new FormData();
            formData.append('video', videoFile.files[0]);

            try {
                // Use the new endpoint for extraction
                const response = await fetch('/upload_for_extraction', {
                    method: 'POST',
                    body: formData,
                });

                const result = await response.json();

                if (response.ok) {
                    uploadStatus.textContent = `Success: ${result.message}`;
                    if(uploadResultData) uploadResultData.textContent = `Landmark data saved on server as: ${result.filename}\nClick link to download.`;
                    if(downloadLink) {
                        downloadLink.href = `/data/${result.filename}`;
                        downloadLink.download = result.filename;
                        downloadLink.style.display = 'block';
                    }
                    if(uploadResultsArea) uploadResultsArea.classList.remove('hidden');
                } else {
                    uploadStatus.textContent = `Error: ${result.error || 'Upload for extraction failed'}`;
                }
            } catch (error) {
                console.error('Upload for extraction error:', error);
                uploadStatus.textContent = `Error: ${error.message}`;
            } finally {
                videoFile.value = ''; // Clear the file input
            }
        });
    }


    // --- Webcam Logic for Extractor ---

    function startFrameProcessingExtractor() {
        if (!videoStreamExtractor || processingIntervalExtractor) {
             console.warn("Extractor: startFrameProcessing called without ready stream or already processing.");
             return;
        }

        status.textContent = 'Capturing frames for extraction...';
        collectedLandmarksExtractor = [];
        if(frameCount) frameCount.textContent = '0';
        if(resultsArea) resultsArea.classList.add('hidden');
        if(webcamDownloadLink) webcamDownloadLink.style.display = 'none';

        processingIntervalExtractor = setInterval(() => {
            if (webcamFeed.readyState >= webcamFeed.HAVE_CURRENT_DATA && webcamFeed.videoWidth > 0) {
                if (canvas.width !== webcamFeed.videoWidth || canvas.height !== webcamFeed.videoHeight) {
                    canvas.width = webcamFeed.videoWidth;
                    canvas.height = webcamFeed.videoHeight;
                }
                ctx.drawImage(webcamFeed, 0, 0, canvas.width, canvas.height);
                const frameData = canvas.toDataURL('image/jpeg', 0.8);
                // Emit to the new server event for extraction
                socket.emit('process_frame_for_extraction', frameData);
            }
        }, 1000 / frameRateExtractor);
    }

    async function handleStartClickExtractor() {
        if(startButton) startButton.disabled = true;
        status.textContent = 'Starting webcam for extraction...';

        try {
            if (!videoStreamExtractor) {
                videoStreamExtractor = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                webcamFeed.srcObject = videoStreamExtractor;
                await new Promise((resolve) => {
                    webcamFeed.onloadedmetadata = () => {
                        resolve();
                    };
                });
            }
            
            canvas.width = webcamFeed.videoWidth;
            canvas.height = webcamFeed.videoHeight;
            
            startFrameProcessingExtractor();
            if(stopButton) stopButton.disabled = false;

        } catch (err) {
            console.error("Extractor: Error accessing webcam:", err);
            status.textContent = `Error: ${err.message}`;
            if (videoStreamExtractor) {
                videoStreamExtractor.getTracks().forEach(track => track.stop());
                videoStreamExtractor = null;
                webcamFeed.srcObject = null;
            }
            if(startButton) startButton.disabled = false;
            if(stopButton) stopButton.disabled = true;
        }
    }

    function handleStopClickExtractor() {
        if(stopButton) stopButton.disabled = true;
        if(startButton) startButton.disabled = false;

        if (processingIntervalExtractor) {
            clearInterval(processingIntervalExtractor);
            processingIntervalExtractor = null;
        }

        if (videoStreamExtractor) {
            videoStreamExtractor.getTracks().forEach(track => track.stop());
            videoStreamExtractor = null;
            webcamFeed.srcObject = null;
        }

        status.textContent = `Capture stopped. ${collectedLandmarksExtractor.length} frames collected for extraction.`;

        if (collectedLandmarksExtractor.length > 0) {
            if(resultsArea) resultsArea.classList.remove('hidden');
            const jsonData = JSON.stringify(collectedLandmarksExtractor); // Compact JSON
            const blob = new Blob([jsonData], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            if(webcamDownloadLink){
                webcamDownloadLink.href = url;
                const timestamp = new Date().toISOString().replace(/[:\-T\.Z]/g, '');
                webcamDownloadLink.download = `webcam_landmarks_extraction_${timestamp}.json`;
                webcamDownloadLink.style.display = 'block';
            }
        } else {
            if(resultsArea) resultsArea.classList.add('hidden');
            if(webcamDownloadLink) webcamDownloadLink.style.display = 'none';
        }
         collectedLandmarksExtractor = []; // Clear after saving/attempting to save
    }

    // --- SocketIO Event Listeners for Extractor ---
    socket.on('connect', () => {
        console.log('Connected to server (main.js for extractor)');
        status.textContent = 'Ready. Press Start Capture for extraction.';
        if(startButton) startButton.disabled = false;
        if(stopButton) stopButton.disabled = true;
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server (main.js for extractor)');
        status.textContent = 'Disconnected. Please refresh.';
        if (processingIntervalExtractor) handleStopClickExtractor(); // Stop if was capturing
    });

    // Listen to the new server event name
    socket.on('extraction_frame_result', (data) => {
        if (processingIntervalExtractor) { // Only collect if capturing
            if (data.error) {
                 console.error("Extractor Backend Error:", data.error);
            } else if (data.landmarks) {
                collectedLandmarksExtractor.push(data.landmarks);
                if(frameCount) frameCount.textContent = collectedLandmarksExtractor.length.toString();
            }
        }
    });

    // --- Event Listeners (ensure elements exist before adding listeners) ---
    if (startButton) startButton.addEventListener('click', handleStartClickExtractor);
    if (stopButton) stopButton.addEventListener('click', handleStopClickExtractor);

    // --- Initial UI State for Extractor ---
    status.textContent = 'Initializing extractor...';
    if(startButton) startButton.disabled = true;
    if(stopButton) stopButton.disabled = true;
    if(resultsArea) resultsArea.classList.add('hidden');
    if(uploadResultsArea) uploadResultsArea.classList.add('hidden');
    if(downloadLink) downloadLink.style.display = 'none';
    if(webcamDownloadLink) webcamDownloadLink.style.display = 'none';

}); // End DOMContentLoaded