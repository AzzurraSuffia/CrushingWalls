import numpy as np
import constants

def get_bounding_rectangle(rgb_image, landmarks):
    if not landmarks or landmarks is None:
        return None

    height, width, _ = rgb_image.shape

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    bbox_left = int(min(xs) * width)
    bbox_right = int(max(xs) * width)
    bbox_top = int(min(ys) * height)
    bbox_bottom = int(max(ys) * height)
    bbox = np.array([bbox_left, bbox_right, bbox_bottom, bbox_top], dtype=np.int32)# DEBUG

    return bbox

def is_user_ready(frame, landmarks):
    if not landmarks or landmarks is None:
        return False

    # Condition 1. Upper body landmarks are visible
    # left shoulder, right should, left hip, right hip
    torso_landmarks = [landmarks[11], landmarks[12], landmarks[23], landmarks[24]]
    head_landmarks = [landmarks[0]]

    torso_visible = all(lm.visibility > constants.VISIBILITY_THRESHOLD for lm in torso_landmarks)
    head_visible = all(lm.visibility > constants.VISIBILITY_THRESHOLD for lm in head_landmarks)

    all_visible = torso_visible and head_visible

    # Condition2. Upper body centroid is in the center region
    
    h, w, _ = frame.shape

    center_x_min = constants.CENTER_X_MIN * w
    center_x_max = constants.CENTER_X_MAX * w
    center_y_min = constants.CENTER_Y_MIN * h
    center_y_max = constants.CENTER_Y_MAX * h

    xs = [lm.x * w for lm in torso_landmarks]
    ys = [lm.y * h for lm in torso_landmarks]

    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)

    in_center_x = center_x_min <= centroid_x <= center_x_max
    in_center_y = center_y_min <= centroid_y <= center_y_max

    # Result
    return all_visible and in_center_x and in_center_y

def compute_wall_positions(mapping):
    target = (mapping.closing_bbox_right_start - mapping.closing_bbox_left_start) // 2 + mapping.closing_bbox_left_start

    if mapping.close_counter <= mapping.MAX_CLOSE - constants.CLOSED_PAUSE:
        t = mapping.close_counter / (mapping.MAX_CLOSE - constants.CLOSED_PAUSE)
        left = int(mapping.closing_bbox_left_start * (1 - t) + target * t)
        right = int(mapping.closing_bbox_right_start * (1 - t) + target * t)
    else: # hold wall closed for a little
        left = right = int(target)

    return left, right