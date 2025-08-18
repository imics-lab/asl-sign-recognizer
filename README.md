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
*   **Sign Dictionary:**
    *   Lookup an English word and play back its corresponding ASL sign video.

## Setup and Installation

### Prerequisites

* Docker with Compose V2 (`docker compose ...`) installed and running
* Git for cloning the repository

### Installation

1) Clone the repository
```bash
git clone https://github.com/imics-lab/asl-sign-recognizer.git
cd asl-sign-recognizer
```
2) No manual build step needed. The first run (below) will build the image.

## Running the Application

1) Start with Docker Compose (detached)
```bash
docker compose up --build -d
```
This builds the image (first time) and starts the app on `http://localhost:5000`.

2) Open in your browser
* Main: `http://localhost:5000`
* Landmark Extractor: `http://localhost:5000/landmark_extractor`
* Playback Tool: `http://localhost:5000/playback`
* Sign Dictionary: `http://localhost:5000/sign_lookup`

Notes
* Videos for the Sign Dictionary page are served from `static/videos/` and must be named `<videoKey>.mp4` where `<videoKey>` comes from `resources/nslt_2000.json`.
* If you add or change videos, re-run with `--build` (as above) to bake them into the image. For live editing without rebuilds, you can bind-mount your videos (see Development Workflow).

## Development Workflow

Common commands
```bash
# Start (build if needed) and run in background
docker compose up --build -d

# View logs
docker compose logs -f app

# Stop and remove containers
docker compose down
```

Making code or asset changes
* Rebuild after changes to Python/HTML/JS/static assets (copied into the image):
```bash
docker compose up --build -d
```
* Optional: live-edit Sign Lookup videos without rebuilding by bind-mounting your local folder. Add this line under `services.app.volumes` in `docker-compose.yml`:
```yaml
- /absolute/path/to/static/videos:/app/static/videos:ro
```
Then restart with `docker compose up -d`.

## Code layout

```
asl-sign-recognizer/
├── app.py                    # Flask entrypoint
├── server/                   # Lightweight backend helpers
│   ├── __init__.py
│   ├── lookup.py             # Sign Dictionary mapping utilities
│   └── utils.py              # Landmark extraction utilities
├── models/                   # ML models (unchanged)
├── templates/                # HTML templates
├── static/                   # Static files (JS, CSS, videos)
│   └── videos/               # Sign Dictionary videos (<videoKey>.mp4)
├── resources/                # Backend assets (class lists, model weights, json mappings)
└── docker-compose.yml, Dockerfile, requirements.txt, README.md
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

To enable actual sign recognition, a deep learning model (e.g., LSTM, Transformer) needs to be trained on landmark sequences. The `batch_processing` directory contains scripts that can be adapted for:
1.  Extracting landmarks from a dataset like WLASL (using `batch_process_videos.py`).
2.  Preparing these landmarks into a suitable format (e.g., NumPy arrays with padding/truncation using `create_npy_dataset.py` or `create_split_dataset.py`) for training.
