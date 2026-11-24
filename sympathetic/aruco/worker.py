# worker.py developed by Bob Tianqi Wei

import argparse
import base64
import json
import math
import sys
import time
from pathlib import Path


def publish(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def list_cameras():
    if sys.platform == "darwin":
        from AVFoundation import AVCaptureDevice, AVMediaTypeMuxed, AVMediaTypeVideo

        devices = list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo))
        devices.extend(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeMuxed))
        devices.sort(key=lambda device: str(device.uniqueID()))
        return [
            {"id": str(device.uniqueID()), "name": str(device.localizedName()), "source": index}
            for index, device in enumerate(devices)
        ]

    if sys.platform.startswith("linux"):
        import fcntl
        import os
        import struct

        query_capabilities = 0x80685600  # VIDIOC_QUERYCAP
        capture_flags = 0x00000001 | 0x00001000  # Single-plane or multiplane video capture.
        device_capabilities_flag = 0x80000000
        capability_layout = struct.Struct("=16s32s32s6I")
        devices = []
        for device in sorted(Path("/sys/class/video4linux").glob("video[0-9]*")):
            source = "/dev/" + device.name
            try:
                descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
                try:
                    data = fcntl.ioctl(descriptor, query_capabilities, bytes(capability_layout.size))
                finally:
                    os.close(descriptor)
                _, name, _, _, flags, device_flags, *_ = capability_layout.unpack(data)
                capabilities = device_flags if flags & device_capabilities_flag else flags
                if capabilities & capture_flags:
                    label = name.decode("utf-8", "replace").strip("\0 ")
                    devices.append({"id": source, "name": label or device.name, "source": source})
            except OSError:
                continue
        return devices

    return []


def open_camera(cv2, camera_id):
    device = next((entry for entry in list_cameras() if entry["id"] == camera_id), None)
    if device is None:
        raise RuntimeError("The selected camera is unavailable. Refresh the device list.")
    if sys.platform == "darwin":
        from macos_camera import MacCamera

        return MacCamera(camera_id)
    return cv2.VideoCapture(device["source"], cv2.CAP_V4L2)


def track(cv2, dictionary, parts, camera_id, output):
    from hand_markers import HandTracker

    detector = cv2.aruco.ArucoDetector(dictionary)
    camera = open_camera(cv2, camera_id)
    last_seen = {}
    frame_number = 0
    last_frame_time = time.monotonic()
    restarted = False
    hands = HandTracker()
    try:
        if not camera.isOpened():
            raise RuntimeError("Could not open the camera. Check camera access or choose another device.")

        while True:
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                timeout = 5 if frame_number == 0 else 2
                if time.monotonic() - last_frame_time < timeout:
                    time.sleep(0.05)
                    continue
                if restarted:
                    raise RuntimeError("No frames from the selected camera. Check its connection or close other camera apps.")
                camera.release()
                camera = None
                camera = open_camera(cv2, camera_id)
                if not camera.isOpened():
                    raise RuntimeError("Could not reconnect to the selected camera.")
                last_seen.clear()
                last_frame_time = time.monotonic()
                restarted = True
                continue

            last_frame_time = time.monotonic()
            restarted = False
            height, width = frame.shape[:2]
            if max(height, width) > 640:
                scale = 640 / max(height, width)
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
            corners, ids, _ = detector.detectMarkers(frame)
            now = time.monotonic()
            detected = []
            marker_corners = {}
            marker_ids = {}
            if ids is not None:
                for i, outline in enumerate(corners):
                    marker_id = int(ids[i, 0])
                    if marker_id in parts:
                        last_seen[marker_id] = now
                        marker_corners[marker_id] = outline.reshape(4, 2)
                        marker_ids[marker_id] = ids[i:i + 1]
                        top_left, top_right = marker_corners[marker_id][:2]
                        dx, dy = top_right - top_left
                        angle = round(math.degrees(math.atan2(dy, dx))) % 360
                        detected.append({"id": marker_id, "angle": angle})

            associations = hands.matches(cv2, frame, marker_corners)
            actions = {}
            for marker_id, action, origin in associations:
                actions.setdefault(marker_id, set()).add(action)
                if action == "Pointing":
                    cv2.line(frame, tuple(origin.astype(int)), tuple(marker_corners[marker_id].mean(axis=0).astype(int)),
                             (255, 200, 100), 2, cv2.LINE_AA)
            for marker in detected:
                marker_id = marker["id"]
                marker["hand"] = ", ".join(sorted(actions.get(marker_id, ())))
                color = (255, 200, 100) if marker["hand"] else (60, 255, 60)
                cv2.aruco.drawDetectedMarkers(frame, [marker_corners[marker_id].reshape(1, 4, 2)],
                                              ids=marker_ids[marker_id], borderColor=color)

            # Tolerate a brief detection miss without repeatedly muting a visible section.
            visible = [marker_id for marker_id, seen in last_seen.items() if now - seen < 0.16]
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_number += 1
            publish(output, {
                "frame": frame_number,
                "ids": visible,
                "markers": sorted(detected, key=lambda marker: marker["id"]),
                "hand_status": hands.status,
                "preview": base64.b64encode(jpeg).decode("ascii") if ok else "",
            })
            time.sleep(0.045)
    finally:
        hands.close()
        if camera is not None:
            camera.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("devices", "track", "pdf"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("status", type=Path)
    parser.add_argument("--camera", default="")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--paper", default="US Letter")
    parser.add_argument("--size", default="45")
    args = parser.parse_args()

    try:
        if args.mode == "devices":
            publish(args.status, {"devices": list_cameras()})
            return 0

        import cv2

        parts = {entry["id"]: entry["part"] for entry in json.loads(args.manifest.read_text(encoding="utf-8"))}
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if args.mode == "track":
            track(cv2, dictionary, parts, args.camera, args.status)
        else:
            from marker_pdf import PAPER_SIZES, layout_markers, write_pdf

            paper = PAPER_SIZES[args.paper]
            marks = [(marker_id, part, args.size) for marker_id, part in parts.items()]
            write_pdf(args.pdf, paper, dictionary, layout_markers(marks, paper, dictionary))
            publish(args.status, {"ok": True})
    except ModuleNotFoundError as error:
        publish(args.status, {"error": f"Install the ArUco Python dependencies. Missing module: {error.name}."})
        return 1
    except Exception as error:
        publish(args.status, {"error": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
