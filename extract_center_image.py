import os
import cv2
import glob
import random
import argparse
from typing import List
from ultralytics import YOLO
import numpy as np

# --- Helper functions for rotation and coordinate mapping ---

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

def visualize_existing_contours_and_crop(image, contours):
    # Make a copy so we don't scribble on the original
    vis = image.copy()

    if not contours:
        print("No contours provided.")
        return vis, None

    # Draw ALL contours in green
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # Take the largest contour
    c = max(contours, key=cv2.contourArea)

    # Draw the largest contour in blue
    cv2.drawContours(vis, [c], -1, (255, 0, 0), 3)

    # Approximate polygon corners
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    # Draw corner points in red
    for p in approx:
        cv2.circle(vis, tuple(p[0]), 6, (0, 0, 255), -1)

    # Extract bounding box from corner points
    xs = [p[0][0] for p in approx]
    ys = [p[0][1] for p in approx]

    # Bounding rectangle coordinates
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Crop region from original image
    cropped = image[y_min:y_max, x_min:x_max]

    return vis, cropped


def visualize_existing_contours(image, contours):
    # Make a copy so we don't scribble on the original
    vis = image.copy()

    if not contours:
        print("No contours provided.")
        return vis

    # Draw ALL contours in green
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # Take the largest contour
    c = max(contours, key=cv2.contourArea)

    # Draw the largest contour in blue
    cv2.drawContours(vis, [c], -1, (255, 0, 0), 3)

    # Approximate polygon corners
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    # Draw corner points in red
    for p in approx:
        cv2.circle(vis, tuple(p[0]), 6, (0, 0, 255), -1)

    return vis


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

    img2, img3 = visualize_existing_contours_and_crop(orig, contours)
    cv2.imshow("contours", img2)
    pause_to_view()

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

def main():
    img_path = "/Users/patrickryan/Development/machinelearning/yolo-sandbox/geekforgeeks/yolov8_object_detection/images/real_drone_photos/truck/images/truck_warehouse_lighting_2025-10-21T16-43-15-575Z.jpg"
    warped = extract_and_orient_inner_image(img_path)
    cv2.imshow("warped", warped)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()