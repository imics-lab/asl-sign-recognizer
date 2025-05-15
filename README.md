# ASL Sign Recognizer

This web application allows users to get real-time American Sign Language (ASL) sign predictions. Users can either upload a video of a sign or perform a sign in front of their webcam. The system extracts landmarks using MediaPipe and then (currently using a mock model) predicts the English label for the sign.

The application also includes a separate tool for extracting and downloading MediaPipe landmarks (Pose, Left Hand, Right Hand - 225 features) from videos.

## Features

*   **Sign Recognition:**
    *   Upload pre-recorded videos.
    *   Capture signs live via webcam with a 3-second countdown.
    *   Displays Top-N predicted sign labels with confidence scores (using a mock model).
    *   Option to playback the extracted landmarks from the captured/uploaded video.
    *   Automatic trimming of trailing neutral poses from webcam captures.
    *   Padding/truncation of landmark sequences to a fixed length for model input.
*   **Landmark Extraction Tool:**
    *   Upload videos or use webcam to extract and download MediaPipe landmarks (Pose, Left Hand, Right Hand - 225 features per frame) as JSON files.
*   **Playback Tool:**
    *   Visualize previously extracted landmark JSON files.

## Setup and Installation

### Prerequisites

*   Docker installed and running on your system.
*   Git for cloning the repository.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/imics-lab/asl-sign-recognizer.git
    cd asl-sign-recognizer
    ```

2.  **Build the Docker image:**
    From the root of the `asl-sign-recognizer` directory, run:
    ```bash
    docker build -t asl-recognizer-app .
    ```
    *(Changed image tag to `asl-recognizer-app` for consistency).*

## Running the Application

1.  **Run the Docker container:**
    ```bash
    docker run -p 5000:5000 \
           -v "$(pwd)/data:/app/data" \
           -v "$(pwd)/uploads:/app/uploads" \
           --name asl-app-instance \
           asl-recognizer-app
    ```
    *   This command maps port `5000` from the container to your host.
    *   It mounts a `data` directory from your current host path to `/app/data` in the container. This is where processed landmark JSON files will be stored and accessible for download/playback links.
    *   It mounts an `uploads` directory from your current host path to `/app/uploads` in the container. This is for temporary storage of uploaded videos.
    *   `--name asl-app-instance` gives the container a recognizable name.
    *   Use `-d` flag (`docker run -d ...`) to run in detached mode (in the background).

    **Note for PowerShell users on Windows:** Replace `$(pwd)` with `${PWD}`:
    ```powershell
    docker run -p 5000:5000 -v "${PWD}/data:/app/data" -v "${PWD}/uploads:/app/uploads" --name asl-app-instance asl-recognizer-app
    ```

2.  **Access the application:**
    Open your web browser and navigate to:
    ```
    http://localhost:5000
    ```
    You should see the main ASL Sign Recognition page.
    *   Landmark Extractor: `http://localhost:5000/landmark_extractor`
    *   Playback Tool: `http://localhost:5000/playback`

## Development Workflow

### Stopping the Application

To stop the running container:
```bash
docker stop asl-app-instance
```

### Re-running the Application

If the container is stopped, you can restart it with:
```bash
docker start asl-app-instance
```
(No need to `docker run` again unless you removed it or want to change parameters).

### Modifying Code and Rebuilding

If you make changes to the application code (Python, HTML, JS):

1.  **Stop the current container (if running):**
    ```bash
    docker stop asl-app-instance
    ```
2.  **Remove the stopped container:**
    (This is important as `docker run` with `--name` will conflict if an old instance exists)
    ```bash
    docker rm asl-app-instance
    ```
3.  **Rebuild the Docker image:**
    (This incorporates your code changes into the image)
    ```bash
    docker build -t asl-recognizer-app .
    ```
4.  **Run the newly built image:**
    ```bash
    docker run -p 5000:5000 -v "$(pwd)/data:/app/data" -v "$(pwd)/uploads:/app/uploads" --name asl-app-instance asl-recognizer-app
    ```

## Project Structure

```
asl-sign-recognizer/
├── Dockerfile
├── README.md
├── app.py                   # Flask application, routes, SocketIO logic
├── requirements.txt         # Python dependencies
├── utils.py                 # Landmark extraction and video processing utilities
├── resources/
│   └── wlasl_class_list.txt # List of ASL signs and their labels
├── static/
│   └── js/
│       ├── main.js          # JS for landmark_extractor.html
│       ├── recognition.js   # JS for index.html (sign recognition)
│       └── playback.js      # JS for playback.html
├── templates/
│   ├── index.html           # Main sign recognition page
│   ├── landmark_extractor.html # Dedicated landmark extraction tool
│   └── playback.html        # Landmark playback visualization
└── batch_processing/        # Scripts for offline data preparation (not part of running app)
    ├── batch_process_videos.py
    ├── create_npy_dataset.py
    ├── create_split_dataset.py
    └── ...
```

## Future Work / Model Training

The current application uses a **mock model** for sign prediction. To enable actual sign recognition, a deep learning model (e.g., LSTM, Transformer) needs to be trained on landmark sequences. The `batch_processing` directory contains scripts that can be adapted for:
1.  Extracting landmarks from a dataset like WLASL (using `batch_process_videos.py`).
2.  Preparing these landmarks into a suitable format (e.g., NumPy arrays with padding/truncation using `create_npy_dataset.py` or `create_split_dataset.py`) for training.

Once a model is trained (e.g., in PyTorch or TensorFlow/Keras) and saved, `app.py` would need to be updated to load this model and use it for predictions instead of `mock_pytorch_model`.
