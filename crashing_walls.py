import cv2
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import get_bounding_rectangle
from drawing import draw_landmarks_on_image, draw_bounding_rectangle
from filters import ButterworthMultichannel
from body_landmarks import BodyLandmarks

# Constants
fps = 30
cutoff = 3.0
order = 2
resize_frame_width = 640
resize_frame_height = 480

# Creating a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path='models\\pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Creating filter
butterworth_filter = ButterworthMultichannel(len(BodyLandmarks)*3, order, cutoff, btype='lowpass', fs=fps)

# Selecting the video camera as input source
cap = cv2.VideoCapture(0)
print("Processing webcam input.")

# Checking for possible errors
if not cap.isOpened():
    print("Error in opening the video stream.")
    sys.exit()

while True:
    
    # ------------------ Layer 1 ------------------
    # Getting current frame
    success, current_frame = cap.read()
    if not success:
        break
    # image resize 
    # Note: MediaPipe internally resizes to 256x256
    # Leave this resize iff needed to speed other operations
    current_frame = cv2.resize(current_frame, (resize_frame_width, resize_frame_height))
    current_frame = cv2.flip(current_frame, 1) # mirror

    # For each frame, detect landmarks 
    #we change the image format for compatibility (some precious time is wasted)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=current_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.pose_landmarks:
        
        if len(detection_result.pose_landmarks) > 1:
            # Text parameters
            text = "WARNING: Someone is disturbing the game!"
            position = (50, 50)             
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (0, 0, 255)             
            thickness = 2

            cv2.putText(current_frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        else:
            landmarks = detection_result.pose_landmarks # pose world landmarks better for kinetic energy?
            
            #ADD: filter over landmark visibility

            annotated_image = draw_landmarks_on_image(current_frame, landmarks) # DEBUG

            bbox_left, bbox_right, bbox_top, bbox_bottom = get_bounding_rectangle(current_frame, detection_result)

            bounding_rect = np.array([[bbox_left, bbox_top], [bbox_left, bbox_bottom], [bbox_right, bbox_bottom], [bbox_right, bbox_top]], dtype=np.int32)# DEBUG
            annotated_image = draw_bounding_rectangle(annotated_image, bounding_rect) # DEBUG

            cv2.rectangle(annotated_image, (0, 0), (bbox_left, resize_frame_height), (255, 0, 0), -1)       # left wall
            cv2.rectangle(annotated_image, (bbox_right, 0), (resize_frame_width, resize_frame_height), (0, 165, 255), -1)  # right wall

    else:
        annotated_image = current_frame

    cv2.imshow("Crashing Walls", annotated_image)

    if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
        cap.release()
        break

# Closing video capture device
cv2.destroyAllWindows() 
cv2.waitKey(1)