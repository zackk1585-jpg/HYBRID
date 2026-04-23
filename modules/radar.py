import cv2
import numpy as np
import mss

class Radar:
    def __init__(self):
        self.sct = mss.mss()

        self.monitor = {
            "left": 0,
            "top": 0,
            "width": 800,
            "height": 600
        }

        self.threshold = 85
        self.min_area = 10
        
        # Morphological kernel for noise reduction
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def grab(self):
        """Capture screen and convert color space"""
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def detect(self, frame):
        """Detect fish using enhanced preprocessing"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Threshold
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup: remove small noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Size filtering
            area = w * h
            if area > self.min_area:
                # Reject extremely elongated artifacts (noise)
                aspect_ratio = w / max(h, 1)
                if aspect_ratio > 6 or aspect_ratio < 0.17:
                    continue
                
                # Require minimum compactness
                contour_area = cv2.contourArea(c)
                if contour_area > 0:
                    compactness = area / contour_area
                    if compactness < 0.4:  # Too thin/fragmented
                        continue
                
                detections.append((int(x), int(y), int(w), int(h)))

        return detections

    def show(self, frame, blobs, intersections, class_counts=None):
        """Display detections with class coloring"""
        vis = frame.copy()
        
        # Color map for fish classes
        class_colors = [
            (255, 0, 0),    # Class 0: Blue
            (0, 255, 0),    # Class 1: Green
            (0, 255, 255),  # Class 2: Yellow
            (255, 0, 255),  # Class 3: Magenta
            (255, 165, 0)   # Class 4: Orange
        ]

        # Draw bounding boxes with class color
        for b in blobs:
            color = class_colors[min(b.cls, 4)]
            cv2.rectangle(vis, (b.x, b.y), (b.x+b.w, b.y+b.h), color, 2)
            
            # Add class label
            cv2.putText(vis, f"C{b.cls}", (b.x, b.y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw intersection points (targeting circles) - larger, clearer
        for (x, y) in intersections:
            cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)  # Red filled circle
            cv2.circle(vis, (x, y), 10, (0, 0, 255), 2)  # Red outline

        # Display class counts if available
        if class_counts:
            text = f"Fish: C0:{class_counts[0]} C1:{class_counts[1]} C2:{class_counts[2]} C3:{class_counts[3]} C4:{class_counts[4]}"
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        # Display threshold info
        cv2.putText(vis, f"Threshold: {int(self.threshold)} | MinArea: {self.min_area}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("HYBRID Radar", vis)

        return cv2.waitKey(1) & 0xFF == ord('q')
