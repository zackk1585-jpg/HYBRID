class Blob:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
        self.cls = 0


def update_blobs(detections):
    blobs = []

    for d in detections:
        if len(d) < 4:
            continue

        x, y, w, h = d

        if w < 6 or h < 6:
            continue

        blobs.append(Blob(x, y, w, h))

    if not blobs:
        return [], [], []

    # classify by size
    blobs.sort(key=lambda b: b.area)

    n = len(blobs)

    for i, b in enumerate(blobs):
        ratio = i / max(1, n)

        if ratio < 0.2:
            b.cls = 0
        elif ratio < 0.4:
            b.cls = 1
        elif ratio < 0.6:
            b.cls = 2
        elif ratio < 0.8:
            b.cls = 3
        else:
            b.cls = 4

    # simple intersections
    intersections = []

    for i in range(len(blobs)):
        for j in range(i+1, len(blobs)):
            b1 = blobs[i]
            b2 = blobs[j]

            if abs(b1.x - b2.x) < 80 and abs(b1.y - b2.y) < 80:
                ix = (b1.x + b2.x) // 2
                iy = (b1.y + b2.y) // 2
                intersections.append((ix, iy))

    counts = [0]*5
    for b in blobs:
        counts[b.cls] += 1

    return blobs, intersections, counts