import cv2
import random
from ultralytics import YOLO
import time

def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))


def main():
    print("Hello from yolov8-object-detection!")

    """
    yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt
    """
    yolo = YOLO("yolo11s.pt")

    video_path = "./videos/sample.mp4"
    videoCap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret, frame = videoCap.read()
        if not ret:
            break
        results = yolo.track(frame, stream=True)

        for result in results:
            class_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cls = int(box.cls[0])
                    class_name = class_names[cls]

                    conf = float(box.conf[0])

                    colour = getColours(cls)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                    cv2.putText(frame, f"{class_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colour, 2)

        if frame_count < 20:
            cv2.imshow(winname="Yolo Output", mat=frame)
            cv2.waitKey(10)
            time.sleep(0.5)
        else:
            break

        frame_count += 1

    videoCap.release()




if __name__ == "__main__":
    main()
