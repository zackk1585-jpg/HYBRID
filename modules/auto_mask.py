import cv2
import numpy as np

class AutoMask:
    def __init__(self):
        self.target_min = 6
        self.target_max = 15
        self.class_targets = [2, 3, 4, 3, 2]  # Per-class targets: small→large
        self.adjustment_speed = 0.5  # Slower, more stable adjustments
        self.min_threshold = 20
        self.max_threshold = 100

    def update(self, radar, detections, class_counts=None):
        """Smart auto-tuning based on detection count and fish classes"""
        
        count = len(detections)
        
        # If we have class data, use it for smarter tuning
        if class_counts and len(class_counts) == 5:
            # Check if class distribution is balanced
            class_imbalance = sum(abs(class_counts[i] - self.class_targets[i]) 
                                for i in range(5))
            
            # If imbalanced but within count range, fine-tune threshold more carefully
            if self.target_min <= count <= self.target_max and class_imbalance > 5:
                # Slight threshold adjustment to balance classes
                if max(class_counts) > min(class_counts) * 2:
                    radar.threshold = min(self.max_threshold, 
                                        int(radar.threshold + 0.5))
                return
        
        # Too many detections → tighten
        if count > self.target_max:
            radar.threshold = min(self.max_threshold, 
                                int(radar.threshold + self.adjustment_speed))
            radar.min_area = int(min(100, radar.min_area + 10))
        
        # Too few detections → loosen
        elif count < self.target_min:
            radar.threshold = max(self.min_threshold, 
                                int(radar.threshold - self.adjustment_speed))
            radar.min_area = max(5, int(radar.min_area - 5))
        
        # In range - fine tune
        else:
            # Micro-adjust threshold for stability
            if count > self.target_max * 0.9:
                radar.threshold = min(self.max_threshold, 
                                    int(radar.threshold + 0.3))
            elif count < self.target_min * 1.1:
                radar.threshold = max(self.min_threshold, 
                                    int(radar.threshold - 0.3))
        
        # Clamp values
        radar.threshold = max(self.min_threshold, min(self.max_threshold, radar.threshold))
        radar.min_area = max(5, min(100, radar.min_area))
