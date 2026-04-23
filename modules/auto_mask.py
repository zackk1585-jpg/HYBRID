class AutoMask:
    def __init__(self):
        self.target_min = 8
        self.target_max = 20

    def update(self, radar, detections):
        count = len(detections)

        # too noisy → tighten
        if count > self.target_max:
            radar.threshold = min(100, radar.threshold + 1)
            radar.min_area += 50

        # too empty → loosen
        elif count < self.target_min:
            radar.threshold = max(1, radar.threshold - 1)
            radar.min_area = max(50, radar.min_area - 50)

        # clamp
        radar.min_area = max(50, radar.min_area)