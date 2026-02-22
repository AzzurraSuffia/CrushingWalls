import numpy as np
import cv2

# DEBUG
def draw_bounding_rectangle(mask, rectangle, color=(0, 255, 0), thickness=2, fill=False):
    if rectangle is None or rectangle.shape != (4, 2):
        return mask

    # Ensure integer coordinates
    pts = np.array([[int(rectangle[i, 0]), int(rectangle[i, 1])] for i in range(4)], dtype=np.int32).reshape((-1, 1, 2))

    # Make a copy to avoid modifying original mask
    mask_copy = mask.copy()

    if fill:
        cv2.fillPoly(mask_copy, [pts], color)
    else:
        cv2.polylines(mask_copy, [pts], isClosed=True, color=color, thickness=thickness)

    return mask_copy

# DEBUG
def draw_landmarks_on_image(rgb_image, landmarks, draw_connections=True):
    annotated_image = rgb_image.copy()

    # Check if any poses were detected
    if landmarks:
        for pose_landmarks in landmarks:
            # Convert normalized landmarks to pixel coordinates
            h, w, _ = annotated_image.shape
            landmark_points = []
            for lm in pose_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_points.append((x, y))
                cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)  # small green dot

            # Draw connections if requested
            if draw_connections:
                # Define the typical pose connections (subset)
                POSE_CONNECTIONS = [
                    (0, 1), (1, 2), (2, 3), (3, 7),  # Head/neck
                    (0, 4), (4, 5), (5, 6), (6, 8),  # Other side
                    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # torso/limbs
                ]
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                        cv2.line(annotated_image, landmark_points[start_idx], landmark_points[end_idx], (0, 255, 255), 2)

    return annotated_image