import argparse
import cv2
import random
from ultralytics import YOLO
import time
import math

def get_image_contours(img):

    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold to isolate dark regions
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours



def crop_inner_region(image, contours):
    # Validate inputs
    if image is None or not contours:
        return None

    # Take the largest contour
    c = max(contours, key=cv2.contourArea)

    # Approximate polygon corners
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    # Ensure we got enough corner points (e.g., >= 3)
    if approx is None or len(approx) < 3:
        return None

    # Extract bounding box extents from corner coords
    xs = [p[0][0] for p in approx]
    ys = [p[0][1] for p in approx]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Validate sane bounds
    if x_min >= x_max or y_min >= y_max:
        return None

    # Crop from original
    cropped = image[y_min:y_max, x_min:x_max]

    # Make sure crop isn't empty
    if cropped.size == 0:
        return None

    return cropped

def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return (0,255,0)
    # return tuple(random.randint(0, 255) for _ in range(3))


def rotate_image_bound(image, angle_degrees: float):
    """Rotate image by angle (degrees) expanding the canvas to keep full content."""
    (h, w) = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # Compute new bounding dimensions to avoid cropping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated


def main(degrees: float = 0.0, conf_thres: float = 0.4, model_path: str = "yolo11m.pt", video_path: str = "./videos/dexi_camera_all_classes.mp4"):
    print("Hello from yolov-object-detection!")

    """
    yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt
    """
    yolo = YOLO(model_path)

    videoCap = cv2.VideoCapture(video_path)

    if not videoCap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = 0

    while True:
        ret, frame = videoCap.read()
        if not ret:
            break

        cv2.imshow(winname="Raw Video Output", mat=frame)

        # try to get just the portion of the image with the
        # object detection region
        contours = get_image_contours(frame)
        cropped = crop_inner_region(frame, contours)
        if cropped is None:
            continue

        frame = cropped
        cv2.imshow(winname="Cropped Video Output", mat=frame)
        cv2.moveWindow("Cropped Video Output", 200, 300)

        # Determine rotation step and max iterations to cover ~360 degrees
        step = abs(degrees) % 360.0
        if step == 0.0:
            max_steps = 1
        else:
            max_steps = int(math.ceil(360.0 / step))

        found = False
        current_img = frame

        for i in range(max_steps):
            if step != 0.0:
                # Compute cumulative angle from the original frame to avoid canvas growth
                angle = (i + 1) * step
                current_img = rotate_image_bound(frame, angle)
            else:
                current_img = frame

            # cv2.imshow(winname="Rotated Yolo Output", mat=current_img)

            # Run YOLO on the currently rotated image
            results = yolo.predict(source=current_img, verbose=False)

            if not results:
                continue

            result = results[0]
            class_names = result.names

            # If there are boxes and any passes threshold, draw and break
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= conf_thres:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        class_name = class_names[cls]
                        if class_name in ["car", "bird", "cat", "dog", "motorcycle", "truck"]:
                            colour = getColours(cls)
                            cv2.rectangle(current_img, (x1, y1), (x2, y2), colour, 2)
                            cv2.putText(current_img, f"{class_name} {conf:.2f}", (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                            found = True
                            cv2.imshow(winname="Detected Rotated Yolo Output", mat=current_img)
                            cv2.moveWindow("Detected Rotated Yolo Output", 0, 300)
                            break
                        else:
                            tmp_img = current_img.copy()
                            colour = getColours(cls)
                            cv2.rectangle(tmp_img, (x1, y1), (x2, y2), colour, 2)
                            cv2.putText(tmp_img, f"{class_name} {conf:.2f}", (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                            cv2.imshow(winname="All Detections Yolo Output", mat=tmp_img)
                            cv2.moveWindow("All Detections Yolo Output", 0, 600)

                if found:
                    # Optional: Log the angle where detection was found
                    # print(f"Detection at ~{total_angle % 360:.1f} degrees")
                    break

        # Show the chosen orientation for this frame (detected or last tried)

        # Small wait to allow imshow to render; 1 ms is typical for video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    videoCap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO on video frames with optional rotation.")
    parser.add_argument("--degrees", type=float, default=30.0, help="Degrees to rotate each frame before inference (counter-clockwise).")
    parser.add_argument("--conf", type=float, default=0.64, help="Confidence threshold for displaying detections.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to YOLO model file.")
    parser.add_argument("--video", type=str, default="./videos/dexi_camera_all_classes.mp4", help="Path to input video file.")
    args = parser.parse_args()

    main(degrees=args.degrees, conf_thres=args.conf, model_path=args.model, video_path=args.video)
