import numpy as np
import cv2

import constants

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

def draw_energy_bar(frame, energy, threshold):

    h, w, _ = frame.shape
    bar_height = 25
    margin = 20
    bar_width = w - 2 * margin
    x_start = margin
    y_start = h - bar_height - margin


    # Background bar
    cv2.rectangle(frame,
                  (x_start, y_start),
                  (x_start + bar_width, y_start + bar_height),
                  (50, 50, 50),
                  -1)

    # Energy bar
    filled_width = int(bar_width * energy)
    color = (0, 200, 0) if energy > threshold else (0, 0, 255)

    cv2.rectangle(frame,
                  (x_start, y_start),
                  (x_start + filled_width, y_start + bar_height),
                  color,
                  -1)

    # Threshold line
    threshold_x = x_start + int(bar_width * threshold)
    cv2.line(frame,
             (threshold_x, y_start),
             (threshold_x, y_start + bar_height),
             (0, 0, 255),
             2)

    # Label text ("Energy")
    text = "Energy"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Position text above the bar
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x_start + (bar_width - text_w) // 2
    text_y = y_start - 5  # slightly above the top of the bar

    # Draw text in white with black outline for visibility
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    return frame

def draw_message(frame, message, 
                 position = (50, 50), font = cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale = 0.8, color = (0, 0, 255), thickness = 2):

    return cv2.putText(frame, message, position, font, font_scale, color, thickness, cv2.LINE_AA)

def draw_walls(frame, bbox_left, bbox_right, color_left=(255, 0, 0), color_right=(0, 165, 255)):
    output_frame = cv2.rectangle(frame, (0, 0), (bbox_left, constants.RESIZE_H), color_left, -1)       # left wall
    output_frame = cv2.rectangle(frame, (bbox_right, 0), (constants.RESIZE_W, constants.RESIZE_H), color_right, -1)  # right wall
    return output_frame