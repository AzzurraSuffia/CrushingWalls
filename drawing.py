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
def draw_landmarks_on_image(rgb_image, pose_landmarks, draw_connections=True):
    annotated_image = rgb_image.copy()

    # Check if any poses were detected
    if pose_landmarks:
        for landmarks in pose_landmarks:
            # Convert normalized landmarks to pixel coordinates
            h, w, _ = annotated_image.shape
            landmark_points = []
            for lm in landmarks:
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

def overlay_logo(frame, logo, center_x, center_y):
    h_logo, w_logo = logo.shape[:2]

    # Compute ROI coordinates in frame
    y1 = max(center_y - h_logo//2, 0)
    y2 = min(center_y + h_logo//2, frame.shape[0])
    x1 = max(center_x - w_logo//2, 0)
    x2 = min(center_x + w_logo//2, frame.shape[1])

    # Crop the logo if ROI is smaller than logo
    logo_y1 = 0
    logo_y2 = y2 - y1
    logo_x1 = 0
    logo_x2 = x2 - x1

    logo_crop = logo[logo_y1:logo_y2, logo_x1:logo_x2]

    # Separate alpha channel
    if logo_crop.shape[2] == 4:
        logo_rgb = logo_crop[:, :, :3]
        alpha = logo_crop[:, :, 3] / 255.0
    else:
        logo_rgb = logo_crop
        alpha = np.ones((logo_crop.shape[0], logo_crop.shape[1]))

    # Blend
    roi = frame[y1:y2, x1:x2]
    for c in range(3):
        roi[:, :, c] = (alpha * logo_rgb[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)

    frame[y1:y2, x1:x2] = roi
    return frame