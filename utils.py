import numpy as np

def get_bounding_rectangle(rgb_image, detection_result):
    height, width, _ = rgb_image.shape

    xs = [lm.x for lm in detection_result.pose_landmarks[0]]
    ys = [lm.y for lm in detection_result.pose_landmarks[0]]

    bbox_left = int(min(xs) * width)
    bbox_right = int(max(xs) * width)
    bbox_top = int(min(ys) * height)
    bbox_bottom = int(max(ys) * height)

    return bbox_left, bbox_right, bbox_top, bbox_bottom