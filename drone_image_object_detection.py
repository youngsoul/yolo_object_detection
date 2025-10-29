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

def extract_and_orient_inner_image(img_path):
    # Load image
    img = cv2.imread(img_path)
    orig = img.copy()

    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold to isolate dark regions
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Take the largest contour (assumed inner black area)
    c = max(contours, key=cv2.contourArea)

    # Approximate to 4 corners
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    # Ensure we have 4 corners
    if len(approx) != 4:
        # raise ValueError("Did not find a quadrilateral region")
        return None

    # Order corners
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # sorting points: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Compute new width/height
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Destination points for warp
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Perspective warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # Ensure long edge is on top
    h, w = warped.shape[:2]
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped

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

    image_paths = collect_images_from_dir(image_dir)
    if not image_paths:
        print(f"No images found in '{image_dir}'. Supported extensions: jpg, jpeg, png, bmp, tif, tiff, webp")
        return

    if grid_n is None or grid_n < 1:
        print(f"[Warn] Invalid grid size '{grid_n}', using 1 (no tiling).")
        grid_n = 1

    if tile_scale is None or tile_scale <= 0:
        print(f"[Warn] Invalid tile scale '{tile_scale}', using 1.0.")
        tile_scale = 1.0

    # Rotation step validation (degrees); 0 -> disabled
    if tile_rotate_step is None or tile_rotate_step <= 0:
        tile_rotate_step = 0.0
    else:
        # Cap the maximum number of rotations to avoid excessive compute
        approx_iters = int(360.0 / float(tile_rotate_step))
        if approx_iters > 720:
            print(f"[Warn] Rotation step {tile_rotate_step}° implies ~{approx_iters} rotations per tile; this may be very slow.")

    print(f"Loading YOLO model: {model_path}")
    yolo = YOLO(model_path)

    # Get class names map from the model
    class_names = getattr(yolo, 'names', None)
    if class_names is None:
        model_attr = getattr(yolo, 'model', None)
        class_names = getattr(model_attr, 'names', {}) if model_attr is not None else {}

    window_name = "YOLOv8 Image Detection (grid)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Instructions: Press SPACE for next image, 'q' or ESC to quit.")

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"Processing image {idx}/{len(image_paths)}: {img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[Warning] Could not read image: {img_path}")
            continue

        h, w = frame.shape[:2]

        # Optionally draw grid overlay
        if show_grid and grid_n > 1:
            overlay = frame.copy()
            # vertical lines
            for c in range(1, grid_n):
                x = (w * c) // grid_n
                cv2.line(overlay, (x, 0), (x, h), (128, 128, 128), 1)
            # horizontal lines
            for r in range(1, grid_n):
                y = (h * r) // grid_n
                cv2.line(overlay, (0, y), (w, y), (128, 128, 128), 1)
            frame_with_grid = overlay
        else:
            frame_with_grid = frame

        cv2.imshow("grid", frame_with_grid)
        image2 = extract_and_orient_inner_image(img_path)
        if image2 is not None:
            cv2.imshow("Extracted", image2)
        pause_to_view()

        detections = []  # aggregated detections from all tiles

        # Iterate over tiles
        for r in range(grid_n):
            for c in range(grid_n):
                x0 = (w * c) // grid_n
                x1 = (w * (c + 1)) // grid_n
                y0 = (h * r) // grid_n
                y1 = (h * (r + 1)) // grid_n
                tile = frame[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                # Optionally scale the tile before inference
                interp = cv2.INTER_LINEAR if tile_scale >= 1.0 else cv2.INTER_AREA
                if abs(tile_scale - 1.0) > 1e-6:
                    resized_tile = cv2.resize(tile, None, fx=tile_scale, fy=tile_scale, interpolation=interp)
                    scale_used = tile_scale
                else:
                    resized_tile = tile
                    scale_used = 1.0

                # Determine rotation angles
                if tile_rotate_step and tile_rotate_step > 0:
                    # Ensure step is positive float and generate angles [0, 360)
                    step = float(tile_rotate_step)
                    # Limit number of steps to avoid float accumulation beyond 360
                    max_steps = int(np.ceil(360.0 / step))
                    angles = [i * step for i in range(max_steps) if i * step < 360.0]
                else:
                    angles = [0.0]

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
                    results = yolo(rotated_img)
                    if not results:
                        continue
                    result = results[0]
                    # Extract boxes, map back to original scaled-tile coordinates using inverse rotation,
                    # then to original tile coordinates via scale, and finally to global image coordinates.
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                            x1b, y1b, x2b, y2b = map(float, box.xyxy[0])
                            # Corners in rotated image coordinate frame
                            corners_rot = np.array([[x1b, y1b],
                                                    [x2b, y1b],
                                                    [x2b, y2b],
                                                    [x1b, y2b]], dtype=np.float32)
                            # Map corners back to scaled-tile coordinates via inverse affine
                            corners_scaled = apply_affine_to_points(Minv, corners_rot)
                            # Axis-aligned bbox in scaled-tile frame
                            x1s = float(np.min(corners_scaled[:, 0]))
                            y1s = float(np.min(corners_scaled[:, 1]))
                            x2s = float(np.max(corners_scaled[:, 0]))
                            y2s = float(np.max(corners_scaled[:, 1]))
                            # Map to original (unscaled) tile coordinates
                            x1t = x1s / scale_used
                            y1t = y1s / scale_used
                            x2t = x2s / scale_used
                            y2t = y2s / scale_used
                            # Offset by tile origin and clamp to image bounds
                            gx1 = max(0, min(w - 1, int(x0 + x1t)))
                            gy1 = max(0, min(h - 1, int(y0 + y1t)))
                            gx2 = max(0, min(w - 1, int(x0 + x2t)))
                            gy2 = max(0, min(h - 1, int(y0 + y2t)))
                            cls = int(box.cls[0]) if hasattr(box, 'cls') else -1
                            print(f"  tile={r},{c} cls={class_names[cls]}  conf={conf:.2f}")
                            # if the class_names[cls] is any in list

                            if class_names[cls] in ["car", "bird", "cat", "dog", "motorcycle", "truck"]:
                                # check to see if detections already has this class
                                Found = False
                                for d in detections:
                                    if d['cls'] == cls:
                                        # print(f"  duplicate detection: {d}")
                                        Found = True
                                        break
                                if not Found and conf > conf_threshold:
                                    detections.append({
                                        'x1': gx1, 'y1': gy1, 'x2': gx2, 'y2': gy2,
                                        'cls': cls, 'conf': conf
                                    })
                                    found_class_in_rotated_tile = True
                                    break

        # Draw aggregated detections on the full frame (with optional grid overlay)
        draw_detections_aggregated(frame_with_grid, detections, class_names, conf_threshold=conf_threshold)

        # Show the image with detections
        rot_info = f"  rotate_step={tile_rotate_step:.1f}°" if tile_rotate_step and tile_rotate_step > 0 else ""
        title = f"{os.path.basename(img_path)}  ({idx}/{len(image_paths)})  grid={grid_n}x{grid_n}  scale={tile_scale:.2f}{rot_info}"
        cv2.imshow(window_name, frame_with_grid)
        cv2.setWindowTitle(window_name, title)

        # Wait for space (next) or q/esc (quit)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (32, ord(' ')):  # SPACE
                break
            if key in (27, ord('q'), ord('Q')):  # ESC or q to quit
                cv2.destroyAllWindows()
                return
            # Otherwise, keep waiting

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 on all images in a directory and visualize detections with optional N×N tiling, per-tile scaling, and per-tile rotations.")
    parser.add_argument("--image_dir", default="/Users/patrickryan/Development/machinelearning/yolo-sandbox/geekforgeeks/yolov8_object_detection/images/real_drone_photos/dog/images",  help="Path to directory containing images")
    parser.add_argument("--model", default="yolo11x.pt", help="Path to YOLO model weights (e.g., yolo11s.pt)")
    parser.add_argument("--conf", type=float, default=0.6 , help="Confidence threshold for drawing boxes")
    parser.add_argument("--grid", type=int, default=1, help="Grid size N for tiling (NxN). Use 1 for no tiling.")
    parser.add_argument("--show-grid", default=True, action="store_true", help="Overlay the grid lines on the image for visualization.")
    parser.add_argument("--tile-scale", type=float, default=1.0, help="Scale factor applied to each tile before inference (>0).")
    parser.add_argument("--tile-rotate-step", type=float, default=30, help="Degrees step to rotate each tile in [0,360) before inference. 0 disables rotation.")
    args = parser.parse_args()


    process_directory(
        args.image_dir,
        model_path=args.model,
        conf_threshold=args.conf,
        grid_n=args.grid,
        show_grid=args.show_grid,
        tile_scale=args.tile_scale,
        tile_rotate_step=args.tile_rotate_step,
    )


if __name__ == "__main__":
    main()


