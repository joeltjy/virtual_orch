# Sympathetic Orchestra

An interactive web-based orchestra controlled by hand gestures, developed by Bob Tianqi Wei, Shm Garanganao Almeda, Ethan Tam, Dor Abrahamson and Bjoern Hartmann at UC Berkeley, 2024.

## Virtual Orchestra desktop controls

In Virtual Orchestra, open **Workspace → PerformanceView → Orchestra**.

- **Orchestra** shows the existing seating plan.
- **Conducting** launches the local gesture relay when the tab opens. It uses Node.js and loads the
  original camera models from the web. Pick its camera from the menu, then click **Enable Camera**.
  Device names appear after the web view has camera permission. **Open in Browser** provides the same controls in a regular
  browser if the embedded view or camera is unavailable. On Linux the embedded view requires
  WebKitGTK 4.0 and GTK 3 with X11 support.
- **Tangible** lists cameras by name. Choose a camera and click **Start Camera**. A detected ArUco
  marker enables its orchestra part. Removing it releases any notes still sounding for that part.
  The list beside the preview shows each visible marker's angle and whether a hand appears to
  point at it or hold it. Keep the marker pattern visible. MediaPipe processes the same camera feed locally.
  Both cameras keep running when you switch tabs in the Orchestra panel. Use different devices for
  overhead markers and front-facing gestures. With both active, Tangible determines which parts
  can play and Conducting controls tempo, dynamics, and stop/play. **Stop Camera**
  restores manual instrument control. The PDF button prints IDs 1–14 with their assigned names.
  Print at actual size and leave the white margin around each marker.

Set up the Tangible environment with Python 3.12 from the repository root:

```sh
python3.12 -m venv sympathetic/aruco/.venv
sympathetic/aruco/.venv/bin/python -m pip install -r sympathetic/aruco/requirements.txt
```

For an installed app or a different Python environment, set `VIRTUAL_ORCH_ARUCO_PYTHON` to its
Python executable before launching Virtual Orchestra. On macOS, grant camera access when prompted.
On Linux, the camera selector uses available V4L2 video capture devices.

## Overview

Sympathetic Orchestra is a gesture-controlled music application that allows users to conduct a virtual orchestra using hand gestures detected through their webcam. The application uses MediaPipe for hand recognition and gesture detection, p5.js for visualization and audio playback, and Node.js/Express for the web server.

## Features

- **Hand Gesture Recognition**: Control the orchestra using hand gestures detected via webcam
- **Interactive Orchestra**: Multiple instruments that can be controlled individually
- **Real-time Visualization**: Visual feedback synchronized with audio playback
- **Camera Preview**: Optional camera preview to see your gestures
- **Volume Controls**: Adjustable volume sliders for each instrument

## Prerequisites

- Node.js (v16.15.0 or higher recommended)
- npm (v8.10.0 or higher recommended)
- A modern web browser with webcam support (Chrome, Firefox, Safari, Edge)
- Webcam for gesture recognition

## Installation

1. **Clone this repository** to your computer:
   ```bash
   git clone <repository-url>
   cd sympathetic-orchestra
   ```

2. **Check Node.js and npm installation**:
   ```bash
   node -v
   npm -v
   ```

3. **Install dependencies**:
   ```bash
   npm install
   ```

   This will install:
   - `express` - Web server framework
   - `body-parser` - Middleware for parsing request bodies

## Running the Application

1. **Start the server**:
   ```bash
   node App.js
   ```

   You should see the message:
   ```
   Server is running on http://localhost:8000
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:8000/
   ```

3. **Enable the camera** by clicking the "Enable Camera" button to start gesture recognition.

## Available Routes

- **`http://localhost:8000/`** - Main Sympathetic Orchestra application
- **`http://localhost:8000/player`** - Standalone player interface
- **`http://localhost:8000/hands`** - Hand recognition test/demo page

## Project Structure

```
sympathetic-orchestra/
│   README.md
│   App.js                    # Main Express server file
│   package.json              # Node.js dependencies
│
├── public/                   # Web content directory
│   │   index.html            # Main application page
│   │   mySketch.js           # p5.js sketch (visualization & audio)
│   │   recognize_hands.js    # MediaPipe hand gesture recognition
│   │   hands_test.html       # Hand recognition test page
│   │
│   ├── player/               # Standalone player interface
│   │   ├── index.html
│   │   ├── sketch.js
│   │   └── soundfiles/       # Full-length audio files
│   │
│   ├── shorter_soundfiles/   # Shorter audio clips for main app
│   │   └── [instrument].mp3
│   │
│   └── css/
│       └── styles.css        # CSS stylesheet
│
└── app:shared:models/        # MediaPipe model files
    ├── gesture_recognizer.task
    └── hand_landmarker.task
```

## Usage

1. **Enable Camera**: Click "Enable Camera" to start webcam access and gesture recognition
2. **Camera Preview**: Toggle "Camera Preview" to show/hide the webcam feed
3. **Show Sliders**: Click "Show Sliders" to access volume controls for each instrument
4. **Play/Pause**: Use the "Play" button to start or pause the orchestra
5. **Gesture Control**: Use hand gestures to control the orchestra (see gesture recognition documentation)

## Technologies Used

- **Node.js** - Server runtime
- **Express** - Web server framework
- **p5.js** - Creative coding library for visualization and audio
- **MediaPipe** - Hand gesture recognition and tracking
- **HTML5/CSS3** - Frontend structure and styling

## Troubleshooting

- **Error: Cannot find module 'express'** - Run `npm install` to install dependencies
- **Camera not working** - Ensure you've granted camera permissions in your browser
- **Audio not playing** - Check browser audio permissions and ensure sound files are loaded
- **Server port already in use** - Change the port in `App.js` (line 25) if port 8000 is occupied

## Development

To modify the application:

- **HTML/CSS**: Edit files in the `public/` directory
- **Server routes**: Modify `App.js` to add new routes or change server behavior
- **Visualization**: Edit `public/mySketch.js` for p5.js canvas and audio logic
- **Gesture recognition**: Edit `public/recognize_hands.js` for hand tracking behavior

## License

ISC

## Credits

- **Developers**: Bob Tianqi Wei, Shm Garanganao Almeda, Ethan Tam, Dor Abrahamson, Bjoern Hartmann
- **Institution**: UC Berkeley, 2024
- **Original Processing Version**: `Original_Sympathetic_Orchestra.pde`

## References

- [Express.js Documentation](https://expressjs.com/)
- [p5.js Documentation](https://p5js.org/)
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [Node.js Documentation](https://nodejs.org/)
