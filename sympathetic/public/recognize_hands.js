//Sympathetic Orchestra 
//by Bob Tianqi Wei, Shm Garanganao Almeda, Ethan Tam, Dor Abrahamson and Bjoern Hartmann
//UC Berkeley, 2024

// This is the code for the hand gesture recognition model.

import {
  GestureRecognizer,
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

window.sharedData = {
  leftHandCursorX: 0,
  leftHandCursorY: 0,
  rightHandCursorX: 0,
  rightHandCursorY: 0,
  leftGestureData: {
    gestureName: "",
    gestureScore: 0,
    handedness: ""
  },
  rightGestureData: {
    gestureName: "",
    gestureScore: 0,
    handedness: ""
  },
  // Added for Sympathetic Virtual Orchestra (conduct.js): all 21 landmarks per hand and when
  // each hand was last seen (performance.now() ms), so stale gestures can be ignored.
  leftLandmarks: null,
  rightLandmarks: null,
  leftSeenMs: -1,
  rightSeenMs: -1
};

document.addEventListener("DOMContentLoaded", () => {
  let gestureRecognizer;
  let handLandmarker;
  let cameraTimer;
  let lastVideoTime = -1;

  const video = document.getElementById("webcam");
  const enableCameraButton = document.getElementById("enableCameraButton");
  const cameraPreviewButton = document.getElementById("cameraPreviewButton");
  const videoContainer = document.getElementById("videoContainer");
  const cameraSelect = document.getElementById("cameraSelect");
  const cameraStatus = document.getElementById("cameraStatus");
  let cameraPreviewVisible = false;

  // Initialize the gesture recognizer
  const createGestureRecognizer = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        delegate: "GPU",
      },
      numHands: 2,
      runningMode: "VIDEO"
    });
    // Enable button when models are loaded
    if (gestureRecognizer && handLandmarker) {
      enableCameraButton.innerText = "Enable Camera";
      enableCameraButton.disabled = false;
    }
  };
  const reportModelError = (error) => {
    enableCameraButton.innerText = "Hand models unavailable";
    console.error(error);
  };
  createGestureRecognizer().catch(reportModelError);

  // Initialize the hand landmarker
  const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
    // Enable button when models are loaded
    if (gestureRecognizer && handLandmarker) {
      enableCameraButton.innerText = "Enable Camera";
      enableCameraButton.disabled = false;
    }
  };
  createHandLandmarker().catch(reportModelError);
  
  // Camera Preview toggle functionality
  cameraPreviewButton.addEventListener("click", () => {
    cameraPreviewVisible = !cameraPreviewVisible;
    if (cameraPreviewVisible) {
      videoContainer.classList.add("visible");
      cameraPreviewButton.innerText = "Hide Preview";
    } else {
      videoContainer.classList.remove("visible");
      cameraPreviewButton.innerText = "Camera Preview";
    }
  });

  const reportCameraStatus = (text) => {
    if (cameraStatus) cameraStatus.textContent = text;
    else if (text) console.warn(text);
  };

  const refreshCameras = async () => {
    if (!cameraSelect || !navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices())
        .filter(({ kind, deviceId }) => kind === "videoinput" && deviceId);
      const selected = video.srcObject?.getVideoTracks()[0]?.getSettings().deviceId ?? cameraSelect.value;
      cameraSelect.replaceChildren(new Option("Default camera", ""));
      devices.forEach((device, i) => {
        cameraSelect.add(new Option(device.label || `Camera ${i + 1}`, device.deviceId));
      });
      cameraSelect.value = devices.some(({ deviceId }) => deviceId === selected) ? selected : "";
    } catch (error) {
      console.warn("Could not list conducting cameras", error);
    }
  };
  refreshCameras();
  navigator.mediaDevices?.addEventListener("devicechange", refreshCameras);

  const stopCamera = () => {
    clearInterval(cameraTimer);
    video.srcObject?.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
    window.sharedData.leftSeenMs = window.sharedData.rightSeenMs = -1;
    if (cameraSelect) cameraSelect.disabled = false;
    enableCameraButton.innerText = "Enable Camera";
    window.onConductingCameraStopped?.();
  };

  enableCameraButton.addEventListener("click", async () => {
    reportCameraStatus("");
    if (video.srcObject) {
      stopCamera();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      reportCameraStatus("Camera access is unavailable in this web view.");
      return;
    }

    enableCameraButton.disabled = true;
    if (cameraSelect) cameraSelect.disabled = true;
    try {
      const selected = cameraSelect?.value;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: selected ? { deviceId: { exact: selected } } : true,
      });
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      lastVideoTime = -1;
      enableCameraButton.innerText = "Disable Camera";
      stream.getVideoTracks()[0].addEventListener("ended", () => {
        if (video.srcObject !== stream) return;
        stopCamera();
        reportCameraStatus("Camera disconnected. Select a camera and try again.");
        refreshCameras();
      });
      // Capture and gesture dispatch continue even when the drawing surface is hidden.
      cameraTimer = setInterval(predictWebcam, 33);
      refreshCameras();
    } catch (error) {
      stopCamera();
      reportCameraStatus(error.name === "NotAllowedError" ? "Allow camera access and try again."
        : "Could not open this camera. Choose a different camera and try again.");
      refreshCameras();
    } finally {
      enableCameraButton.disabled = false;
    }
  });

  const predictWebcam = () => {
    if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
    const nowInMs = performance.now();
    lastVideoTime = video.currentTime;
    try {
      const results = gestureRecognizer.recognizeForVideo(video, nowInMs);
      const handLandmarkerResults = handLandmarker.detectForVideo(
        video,
        nowInMs
      );

      if (handLandmarkerResults.landmarks && handLandmarkerResults.landmarks.length > 0) {
        handLandmarkerResults.landmarks.forEach((landmarks, index) => {

          const eighthLandmark = landmarks[8];
          if (handLandmarkerResults.handednesses[index][0].categoryName === "Right") {
            window.sharedData.leftHandCursorX = eighthLandmark.x;
            window.sharedData.leftHandCursorY = eighthLandmark.y;
            window.sharedData.leftLandmarks = landmarks;
            window.sharedData.leftSeenMs = performance.now();
          } else {
            window.sharedData.rightHandCursorX = eighthLandmark.x;
            window.sharedData.rightHandCursorY = eighthLandmark.y;
            window.sharedData.rightLandmarks = landmarks;
            window.sharedData.rightSeenMs = performance.now();
          }
        });
      }
      //console.log(results);

      if (results.gestures.length > 0) {
        results.gestures.forEach((gesture, index) => {
          const categoryName = gesture[0].categoryName;
          const categoryScore = parseFloat(
            gesture[0].score * 100
          ).toFixed(2);
          const handedness = results.handednesses[index][0].displayName;
          
          // Gesture output element removed - no longer displaying gesture info on screen

          if (handedness == "Right") {
            window.sharedData.leftGestureData.gestureName = categoryName;
            window.sharedData.leftGestureData.gestureScore = categoryScore;
            window.sharedData.leftGestureData.handedness = handedness;
          } else if (handedness=="Left") {
            window.sharedData.rightGestureData.gestureName = categoryName;
            window.sharedData.rightGestureData.gestureScore = categoryScore;
            window.sharedData.rightGestureData.handedness = handedness;
          }
        });
      }
      window.onConductingHands?.(nowInMs);
    } catch (error) {
      stopCamera();
      reportCameraStatus("Hand tracking stopped. Enable the camera to retry.");
      console.error(error);
    }
  };
});

