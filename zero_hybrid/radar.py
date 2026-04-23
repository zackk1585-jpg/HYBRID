import cv2
import numpy as np
import mss

class Radar:
    def __init__(self):
        self.sct = mss.mss()

        # RIGHT HALF OF SCREEN
        monitor_full = self.sct.monitors[1]
        self.monitor = {
            "left": monitor_full["width"] // 2,
            "top": 0,
            "width": monitor_full["width"] // 2,
            "height": monitor_full["height"]
        }

        # sliders initial
        self.threshold = 90
        self.min_area = 20

        self._init_windows()

    # -------------------------
    # UI
    # -------------------------
    def _init_windows(self):
        cv2.namedWindow("mask")
        cv2.namedWindow("radar")

        cv2.createTrackbar("threshold", "mask", self.threshold, 255, lambda x: None)
        cv2.createTrackbar("min_area", "mask", self.min_area, 500, lambda x: None)

        cv2.createTrackbar("threshold", "radar", self.threshold, 255, lambda x: None)
        cv2.createTrackbar("min_area", "radar", self.min_area, 500, lambda x: None)

    def _read_sliders(self):
        self.threshold = cv2.getTrackbarPos("threshold", "mask")
        self.min_area = cv2.getTrackbarPos("min_area", "mask")

    # -------------------------
    # CAPTURE
    # -------------------------
    def grab(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # -------------------------
    # DETECTION
    # -------------------------
    def detect(self, frame):
        self._read_sliders()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

        # smooth to reduce multi-dot issue
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if w * h > self.min_area:
                cx = x + w // 2
                cy = y + h // 2

                detections.append((cx, cy, w, h))  # center-based

        return detections, mask

    # -------------------------
    # DISPLAY
    # -------------------------
    def show(self, frame, detections, mask):
        vis = frame.copy()

        for (cx, cy, w, h) in detections:
            # single dot per fish
            cv2.circle(vis, (cx, cy), 4, (0,255,0), -1)

            # optional box
            cv2.rectangle(
                vis,
                (cx - w//2, cy - h//2),
                (cx + w//2, cy + h//2),
                (255,0,0),
                1
            )

        cv2.imshow("mask", mask)
        cv2.imshow("radar", vis)

        return cv2.waitKey(1) & 0xFF == ord('q')