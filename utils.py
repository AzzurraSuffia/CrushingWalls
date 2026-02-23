import numpy as np
import cv2

def get_bounding_rectangle(rgb_image, detection_result):
    height, width, _ = rgb_image.shape

    xs = [lm.x for lm in detection_result.pose_landmarks[0]]
    ys = [lm.y for lm in detection_result.pose_landmarks[0]]

    bbox_left = int(min(xs) * width)
    bbox_right = int(max(xs) * width)
    bbox_top = int(min(ys) * height)
    bbox_bottom = int(max(ys) * height)

    return bbox_left, bbox_right, bbox_top, bbox_bottom

def is_user_ready(frame, detection_results):
    if not detection_results.pose_landmarks: # nobody detected
        return False
    
    landmarks = detection_results.pose_landmarks[0]

    # Condition 1. Upper body landmarks are visible
    VISIBILITY_THRESHOLD = 0.5

    # left shoulder, right should, left hip, right hip
    torso_landmarks = [landmarks[11], landmarks[12], landmarks[23], landmarks[24]]
    head_landmarks = [landmarks[0]]

    torso_visible = all(lm.visibility > VISIBILITY_THRESHOLD for lm in torso_landmarks)
    head_visible = all(lm.visibility > VISIBILITY_THRESHOLD for lm in head_landmarks)

    all_visible = torso_visible and head_visible

    # Condition 2. Upper body centroid is in the center region
    h, w, _ = frame.shape

    center_x_min = 0.35 * w
    center_x_max = 0.65 * w
    center_y_min = 0.25 * h
    center_y_max = 0.75 * h

    xs = [lm.x * w for lm in torso_landmarks]
    ys = [lm.y * h for lm in torso_landmarks]

    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)

    in_center_x = center_x_min <= centroid_x <= center_x_max
    in_center_y = center_y_min <= centroid_y <= center_y_max

    # Result
    return all_visible and in_center_x and in_center_y