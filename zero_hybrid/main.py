from modules.radar import Radar

def main():
    radar = Radar()

    print("DEBUG MODE (Q = STOP, NO FIRING)")

    while True:
        frame = radar.grab()

        detections, mask = radar.detect(frame)

        if radar.show(frame, detections, mask):
            break


if __name__ == "__main__":
    main()