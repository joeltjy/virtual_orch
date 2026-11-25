# hand_markers.py developed by Bob Tianqi Wei

from queue import Empty, SimpleQueue
from threading import Thread

import numpy as np


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def ray_hit(origin, direction, corners):
    distances = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        edge = end - start
        divisor = cross(direction, edge)
        if abs(divisor) <= 1e-6:
            continue
        offset = start - origin
        distance = cross(offset, edge) / divisor
        position = cross(offset, direction) / divisor
        if distance >= 0 and 0 <= position <= 1:
            distances.append(distance)
    return min(distances, default=float("inf"))


def match_hand(cv2, points, markers, reach):
    """Return a visible marker and the inferred interaction in image coordinates."""
    wrist, thumb, index = points[[0, 4, 8]]
    palm = max(np.linalg.norm(points[9] - wrist), np.linalg.norm(points[17] - points[5]))
    if palm < 5:
        return None

    grips = []
    for marker_id, corners in markers.items():
        thumb_edge = cv2.pointPolygonTest(corners, tuple(thumb), True)
        for tip in points[[8, 12, 16]]:
            spacing = np.linalg.norm(tip - thumb)
            tip_edge = cv2.pointPolygonTest(corners, tuple(tip), True)
            if 0.12 * palm < spacing <= 1.25 * palm and max(abs(thumb_edge), abs(tip_edge)) < 0.26 * palm:
                grips.append((abs(thumb_edge) + abs(tip_edge), marker_id))
    if grips:
        return min(grips)[1], "Holding", (thumb + index) / 2

    finger = index - points[5]
    length = np.linalg.norm(finger)
    segments = np.diff(points[[5, 6, 7, 8]], axis=0)
    if length < 0.42 * palm or length < 0.88 * np.linalg.norm(segments, axis=1).sum():
        return None
    if sum(np.linalg.norm(points[tip] - wrist) < np.linalg.norm(points[tip - 2] - wrist) + 0.12 * palm
           for tip in (12, 16, 20)) < 2:
        return None

    direction = finger / length
    hits = []
    for marker_id, corners in markers.items():
        center = corners.mean(axis=0)
        if cv2.pointPolygonTest(corners, tuple(index), False) >= 0:
            hits.append((0, marker_id))
            continue
        expanded = center + (corners - center) * 1.22
        distance = ray_hit(index, direction, expanded)
        if distance <= reach:
            hits.append((distance, marker_id))
    return (min(hits)[1], "Pointing", index) if hits else None


class HandTracker:
    def __init__(self):
        self.status = "Loading hand tracker…"
        self.ready = SimpleQueue()
        self.landmarker = None
        self.previous_targets = set()
        Thread(target=self.prepare, daemon=True).start()

    def prepare(self):
        try:
            import mediapipe as mp

            self.ready.put((mp.solutions.hands.Hands, ""))
        except (ImportError, AttributeError):
            self.ready.put((None, "Install MediaPipe 0.10.21."))
        except Exception:
            self.ready.put((None, "Hand tracking unavailable."))

    def matches(self, cv2, frame, markers):
        try:
            self.update_model()
            if self.landmarker is None or not markers:
                self.previous_targets.clear()
                return []

            height, width = frame.shape[:2]
            result = self.landmarker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            candidates = {}
            for landmarks in result.multi_hand_landmarks or ():
                points = np.array([(point.x * width, point.y * height) for point in landmarks.landmark])
                match = match_hand(cv2, points, markers, np.hypot(width, height) * 0.65)
                if match is not None:
                    marker_id, action, origin = match
                    candidates[marker_id, action] = origin

            # Require two observations before displaying an association.
            confirmed = [(marker, action, origin) for (marker, action), origin in candidates.items()
                         if (marker, action) in self.previous_targets]
            self.previous_targets = set(candidates)
            return confirmed
        except Exception:
            self.status = "Hand tracking paused."
            self.close()
            return []

    def update_model(self):
        try:
            factory, self.status = self.ready.get_nowait()
        except Empty:
            return
        if factory is not None:
            self.landmarker = factory(max_num_hands=2, model_complexity=0)

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        self.previous_targets.clear()
