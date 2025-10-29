import os
import cv2
import glob
import random
import argparse
from typing import List
from ultralytics import YOLO
import numpy as np

# --- Helper functions for rotation and coordinate mapping ---
import cv2
import numpy as np

show_debug_image_windows = False


def rotate_image_affine(image: np.ndarray, angle_deg: float):
    """Rotate image around its center by angle (degrees) without changing size.
    Returns: rotated_image, M (2x3), Minv (2x3)
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    Minv = cv2.invertAffineTransform(M)
    return rotated, M, Minv


def apply_affine_to_points(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine matrix M to a set of points of shape (N,2)."""
    # Ensure float
    pts = points.astype(np.float32)
    # Append ones for translation
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])  # (N,3)
    M_full = np.vstack([M, [0.0, 0.0, 1.0]])  # (3,3)
    transformed = (M_full @ pts_h.T).T  # (N,3)
    return transformed[:, :2]

# --- Helper functions for rotation and coordinate mapping ---
def pause_to_view():
    print("Instructions: Press SPACE for next image, 'q' or ESC to quit.")
    # Wait for space (next) or q/esc (quit)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (32, ord(' ')):  # SPACE
            break
        if key in (27, ord('q'), ord('Q')):  # ESC or q to quit
            cv2.destroyAllWindows()
            return
        # Otherwise, keep waiting


# def rotate_image_affine(image: np.ndarray, angle_deg: float):
#     """Rotate image around its center by angle (degrees) without changing size.
#     Returns: rotated_image, M (2x3), Minv (2x3)
#     """
#     h, w = image.shape[:2]
#     center = (w / 2.0, h / 2.0)
#     M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
#     rotated = cv2.warpAffine(
#         image,
#         M,
#         (w, h),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=(0, 0, 0),
#     )
#     Minv = cv2.invertAffineTransform(M)
#     return rotated, M, Minv


def apply_affine_to_points(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine matrix M to a set of points of shape (N,2)."""
    # Ensure float
    pts = points.astype(np.float32)
    # Append ones for translation
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])  # (N,3)
    M_full = np.vstack([M, [0.0, 0.0, 1.0]])  # (3,3)
    transformed = (M_full @ pts_h.T).T  # (N,3)
    return transformed[:, :2]


def getColours(cls_num: int):
    """Generate a unique BGR colour tuple for a given class ID."""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))


def collect_images_from_dir(image_dir: str) -> List[str]:
    """Collect image file paths (non-recursive) from the given directory."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    # Sort for deterministic order
    return sorted(paths)


def draw_detections(frame, result, conf_threshold: float = 0.4):
    """Draw bounding boxes and labels on the frame for detections above threshold.
    This version expects a single Ultralytics result object and draws directly.
    """
    class_names = result.names  # mapping from class id to name
    for box in result.boxes:
        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0]) if hasattr(box, "cls") else -1
        class_name = class_names.get(cls, str(cls)) if isinstance(class_names, dict) else (
            class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        )
        colour = getColours(cls)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            lineType=cv2.LINE_AA,
        )


def draw_detections_aggregated(frame, detections, class_names, conf_threshold: float = 0.4):
    """Draw bounding boxes given a list of aggregated detections.
    detections: list of dicts with keys: x1,y1,x2,y2,cls,conf
    class_names: list or dict mapping class id to name
    """
    for det in detections:
        conf = float(det.get("conf", 0.0))
        if conf < conf_threshold:
            print(f"  skipped detection: {det}. {conf} < {conf_threshold}")
            continue
        x1 = int(det["x1"]); y1 = int(det["y1"]); x2 = int(det["x2"]); y2 = int(det["y2"])
        cls = int(det.get("cls", -1))
        if isinstance(class_names, dict):
            class_name = class_names.get(cls, str(cls))
        else:
            class_name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        colour = getColours(cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            lineType=cv2.LINE_AA,
        )


def process_directory(image_dir: str, model_path: str = "yolo11x.pt", conf_threshold: float = 0.4, grid_n: int = 2, show_grid: bool = False, tile_scale: float = 1.0, tile_rotate_step: float = 0.0):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    true_label = image_dir.split("/")[-2]
    correct_label_count = 0
    incorrect_label_count = 0
    total_label_count = 0
    incorrect_label_image_paths = []

    image_paths = collect_images_from_dir(image_dir)
    if not image_paths:
        print(f"No images found in '{image_dir}'. Supported extensions: jpg, jpeg, png, bmp, tif, tiff, webp")
        return

    # Rotation step validation (degrees); 0 -> disabled
    if tile_rotate_step is None or tile_rotate_step <= 0:
        tile_rotate_step = 0.0
    else:
        # Cap the maximum number of rotations to avoid excessive compute
        approx_iters = int(360.0 / float(tile_rotate_step))
        if approx_iters > 720:
            print(f"[Warn] Rotation step {tile_rotate_step}° implies ~{approx_iters} rotations per tile; this may be very slow.")

    print(f"Loading YOLO model: {model_path}")
    yolo = YOLO(model_path, verbose=False)

    # Get class names map from the model
    class_names = getattr(yolo, 'names', None)
    if class_names is None:
        model_attr = getattr(yolo, 'model', None)
        class_names = getattr(model_attr, 'names', {}) if model_attr is not None else {}

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"Processing image {idx}/{len(image_paths)}: {img_path}")
        total_label_count += 1
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[Warning] Could not read image: {img_path}")
            continue

        h, w = frame.shape[:2]

        frame_with_grid = frame

        detections = []  # aggregated detections from all tiles


        # Determine rotation angles
        if tile_rotate_step and tile_rotate_step > 0:
            # Ensure step is positive float and generate angles [0, 360)
            step = float(tile_rotate_step)
            # Limit number of steps to avoid float accumulation beyond 360
            max_steps = int(np.ceil(360.0 / step))
            angles = [i * step for i in range(max_steps) if i * step < 360.0]
        else:
            angles = [0.0]

        resized_tile = frame
        scale_used = 1.0

        for angle in angles:
            found_class_in_rotated_tile = False
            if found_class_in_rotated_tile:
                break
            # Rotate the (scaled) tile if needed
            if abs(angle) > 1e-6:
                rotated_img, M, Minv = rotate_image_affine(resized_tile, angle)
            else:
                rotated_img = resized_tile
                # Identity affine for inverse mapping
                Minv = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0]], dtype=np.float32)

            # cv2.imshow("rotated", rotated_img)
            # pause_to_view()
            # Run inference on the rotated (or original) tile image
            results = yolo(rotated_img, verbose=False)
            if not results:
                continue
            result = results[0]
            # Extract boxes, map back to original scaled-tile coordinates using inverse rotation,
            # then to original tile coordinates via scale, and finally to global image coordinates.
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else -1
                    # print(f"  tile={r},{c} cls={class_names[cls]}  conf={conf:.2f}")
                    # if the class_names[cls] is any in list

                    if class_names[cls] in ["car", "bird", "cat", "dog", "motorcycle", "truck"]:

                        Found = False
                        for d in detections:
                            if d['cls'] == cls:
                                # print(f"  duplicate detection: {d}")
                                Found = True
                                break
                        if not Found and conf > conf_threshold:
                            correct_label_count += 1

                            detections.append({
                                'cls': cls, 'conf': conf, 'class_name': class_names[cls],
                            })
                            found_class_in_rotated_tile = True
                            break

    accuracy = correct_label_count / (total_label_count + 1e-6)
    print(f"Accuracy: {accuracy:.2%}  Correct: {correct_label_count}  Total: {total_label_count}")
    # print(f"Incorrect images: {incorrect_label_image_paths}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 on all images in a directory and visualize detections with optional N×N tiling, per-tile scaling, and per-tile rotations.")
    parser.add_argument("--image_dir", default="/Users/patrickryan/Development/machinelearning/yolo-sandbox/geekforgeeks/yolov8_object_detection/images/real_drone_photos/truck/images",  help="Path to directory containing images")
    parser.add_argument("--model", default="yolo11m.pt", help="Path to YOLO model weights (e.g., yolo11s.pt)")
    parser.add_argument("--conf", type=float, default=0.6 , help="Confidence threshold for drawing boxes")
    parser.add_argument("--tile-rotate-step", type=float, default=30, help="Degrees step to rotate each tile in [0,360) before inference. 0 disables rotation.")
    args = parser.parse_args()


    process_directory(
        args.image_dir,
        model_path=args.model,
        conf_threshold=args.conf,
        tile_rotate_step=args.tile_rotate_step,
    )


if __name__ == "__main__":
    main()


