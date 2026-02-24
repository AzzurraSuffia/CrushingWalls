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

# Computing the first-order derivative
def first_order_derivative(curr_value, prev_value, curr_time, prev_time):
    result = None
    if curr_value is not None and prev_value is not None and curr_time is not None and prev_time is not None:
        dt = curr_time - prev_time
        result = (curr_value - prev_value) / dt
    return result

def compute_kinetic_energy(current_detection_result, previous_detection_result, 
                           prev_time, curr_time,masses=None,
                           apply_filtering=False, velocity_filter=None):

    # Ensure landmarks exist
    if current_detection_result is None or previous_detection_result is None:
        return None
    
    if previous_detection_result.pose_landmarks is None or len(previous_detection_result.pose_landmarks) == 0:
        return None
    
    if current_detection_result.pose_landmarks is None or len(current_detection_result.pose_landmarks) == 0:
        return None

    # Use only the first detected person
    current_landmarks = current_detection_result.pose_landmarks[0]
    previous_landmarks = previous_detection_result.pose_landmarks[0]

    n_landmarks = len(current_landmarks)
    if masses is None:
        masses = np.ones(n_landmarks)

    # Compute velocity vectors for each landmark
    velocities = np.array([
        first_order_derivative(np.array([curr_lm.x, curr_lm.y, curr_lm.z]),
                               np.array([prev_lm.x, prev_lm.y, prev_lm.z]),
                               curr_time, prev_time)
        for curr_lm, prev_lm in zip(current_landmarks, previous_landmarks)
    ])

    # Optional filtering
    if apply_filtering and velocity_filter is not None:
        # Flatten velocities for filtering: (n_points * 3,)
        v_flat = velocities.reshape(-1)
        # Filter one "sample" per channel (vectorized)
        v_f = velocity_filter.filter(v_flat)
        velocities = v_f.reshape(n_landmarks, 3)

    # Compute total kinetic energy
    speed_squared = np.sum(velocities**2, axis=1)
    total_ke = 0.5 * np.sum(masses * speed_squared)

    return total_ke