# macos_camera.py developed by Bob Tianqi Wei

from threading import Event, Lock

import AVFoundation as av
import CoreMedia
import Quartz
import dispatch
import numpy as np
import objc
from Foundation import NSObject


class FrameReceiver(NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
    def init(self):
        self = objc.super(FrameReceiver, self).init()
        if self is not None:
            self.lock = Lock()
            self.ready = Event()
            self.frame = None
            self.error = None
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
        try:
            pixels = CoreMedia.CMSampleBufferGetImageBuffer(sample)
            if pixels is None or Quartz.CVPixelBufferLockBaseAddress(pixels, Quartz.kCVPixelBufferLock_ReadOnly):
                return
            try:
                width = Quartz.CVPixelBufferGetWidth(pixels)
                height = Quartz.CVPixelBufferGetHeight(pixels)
                stride = Quartz.CVPixelBufferGetBytesPerRow(pixels)
                base = Quartz.CVPixelBufferGetBaseAddress(pixels)
                bgra = np.ndarray((height, width, 4), np.uint8, buffer=base.as_buffer(stride * height),
                                  strides=(stride, 4, 1))
                frame = bgra[:, :, :3].copy()
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(pixels, Quartz.kCVPixelBufferLock_ReadOnly)
            with self.lock:
                self.frame = frame
                self.ready.set()
        except Exception as error:
            with self.lock:
                self.error = error
                self.ready.set()


def check_permission():
    access = av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeVideo)
    if access == av.AVAuthorizationStatusAuthorized:
        return
    if access == av.AVAuthorizationStatusNotDetermined:
        answered = Event()
        allowed = False

        def request_completed(granted):
            nonlocal allowed
            allowed = granted
            answered.set()

        av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(av.AVMediaTypeVideo, request_completed)
        if not answered.wait(30):
            raise RuntimeError("Waiting for camera permission timed out. Try Start Camera again.")
        if allowed:
            return

    raise RuntimeError("Camera access is unavailable. Check System Settings → Privacy & Security → Camera.")


class MacCamera:
    def __init__(self, camera_id):
        check_permission()
        device = av.AVCaptureDevice.deviceWithUniqueID_(camera_id)
        if device is None or not device.isConnected():
            raise RuntimeError("The selected camera was disconnected. Refresh the device list.")

        camera_input, error = av.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if camera_input is None:
            raise RuntimeError(error.localizedDescription() if error else "Could not open the camera.")

        self.session = av.AVCaptureSession.alloc().init()
        self.receiver = FrameReceiver.alloc().init()
        self.queue = dispatch.dispatch_queue_create(b"org.virtualorch.aruco-camera", None)
        self.output = av.AVCaptureVideoDataOutput.alloc().init()
        self.output.setAlwaysDiscardsLateVideoFrames_(True)
        self.output.setVideoSettings_({
            Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA,
        })
        self.output.setSampleBufferDelegate_queue_(self.receiver, self.queue)

        if self.session.canSetSessionPreset_(av.AVCaptureSessionPreset640x480):
            self.session.setSessionPreset_(av.AVCaptureSessionPreset640x480)
        if not self.session.canAddInput_(camera_input):
            raise RuntimeError("Could not attach the selected camera.")
        self.session.addInput_(camera_input)
        if not self.session.canAddOutput_(self.output):
            raise RuntimeError("Could not start receiving camera frames.")
        self.session.addOutput_(self.output)
        self.session.startRunning()

    def isOpened(self):
        return self.session.isRunning()

    def read(self):
        if not self.receiver.ready.wait(0.2):
            return False, None
        with self.receiver.lock:
            self.receiver.ready.clear()
            if self.receiver.error is not None:
                raise RuntimeError(f"Could not read a camera frame: {self.receiver.error}")
            frame = self.receiver.frame
        return frame is not None, frame

    def release(self):
        self.session.stopRunning()
        self.output.setSampleBufferDelegate_queue_(None, None)
