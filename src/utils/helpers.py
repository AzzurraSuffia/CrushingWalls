import numpy as np
import config.constants as constants

def get_bounding_rectangle(rgb_image, landmarks):
    """Return bounding box [left, right, bottom, top] of landmarks or None."""

    if not landmarks or landmarks is None:
        return None

    height, width, _ = rgb_image.shape

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    bbox_left = int(min(xs) * width)
    bbox_right = int(max(xs) * width)
    bbox_top = int(min(ys) * height)
    bbox_bottom = int(max(ys) * height)
    bbox = np.array([bbox_left, bbox_right, bbox_bottom, bbox_top], dtype=np.int32)

    return bbox

def is_user_ready(frame, landmarks):
    """Check if user is visible and centered for starting interaction."""

    if not landmarks or landmarks is None:
        return False

    # Condition 1. Upper body landmarks are visible
    # left shoulder, right should, left hip, right hip
    torso_landmarks = [landmarks[11], landmarks[12], landmarks[23], landmarks[24]]
    # nose
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
    """Compute left and right wall positions during closing animation."""

    # Walls joining coordinate
    target = (mapping.closing_bbox_right_start - mapping.closing_bbox_left_start) // 2 + mapping.closing_bbox_left_start

    if mapping.counters.close <= mapping.thresholds.max_close - constants.CLOSED_PAUSE:
        # Compute walls closing positions
        t = mapping.counters.close / (mapping.thresholds.max_close - constants.CLOSED_PAUSE)
        left = int(mapping.closing_bbox_left_start * (1 - t) + target * t)
        right = int(mapping.closing_bbox_right_start * (1 - t) + target * t)
    else: 
        # Hold wall closed for few frames
        left = right = int(target)

    return left, right