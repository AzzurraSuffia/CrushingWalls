import cv2
import sys
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import get_bounding_rectangle, is_user_ready, compute_wall_positions
from drawing import draw_landmarks_on_image, draw_bounding_rectangle, overlay_logo, draw_energy_bar, draw_message, draw_walls, stack_images_horizontal, draw_cv_graph
from filters import ButterworthMultichannel
from body_landmarks import BodyLandmarks
from interaction_fsm import InteractionFSM, State
from ke_processor import KE_Processor
from body_estimator import BodyEstimator
import constants

# NOTES:
# Kinetic energy varies a lot: investigate whether it is normal or not. -> It is normal.

# TODOLIST:
#1. energy bar does not seems responsive. Add a line for average ke on a window eventually.
#2. tune parameters (like max counters and threshold for energy)
#3. Test intrusions

#4. (OPTIONAL) optimize the code by considering only the possible transitions given the state in which the system is and not all of them. 
# Then end, finally :)

#Initialization
ke_display = 0.0

if constants.DEBUG_KE:
    ke_history = deque(maxlen=constants.FPS*constants.PLOT_WINDOW_SECONDS)
    landmarks_history = deque(maxlen=constants.FPS*constants.PLOT_WINDOW_SECONDS)

# Creating a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path=constants.MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# Creating filters
wall_butterworth_filter = ButterworthMultichannel(2, constants.WALL_ORDER, constants.WALL_CUTOFF, btype='lowpass', fs=constants.FPS)
velocity_butterworth_filter = ButterworthMultichannel(len(BodyLandmarks)*3, constants.VELOCITY_ORDER, constants.VELOCITY_CUTOFF, btype='lowpass', fs=constants.FPS)

# Loading logo
logo = cv2.imread(constants.LOGO_PATH, cv2.IMREAD_UNCHANGED)

# Instaciaing objects
mapping = InteractionFSM(max_count=constants.MAX_COUNT, max_close=constants.MAX_CLOSE, threshold=constants.THRESHOLD_KE)
ke_processor = KE_Processor(velocity_butterworth_filter) 
body_estimator = BodyEstimator(constants.ALPHA)

# Selecting the video camera as input source
cap = cv2.VideoCapture(0)
print("Processing webcam input.")

# Checking for possible errors
if not cap.isOpened():
    print("Error in opening the video stream.")
    sys.exit()

while True:
    
    # ------------------ Layer 1 (Input) ------------------
    # Getting current frame
    success, current_frame = cap.read()
    if not success:
        break
    # image resize 
    # Note: MediaPipe internally resizes to 256x256
    # Leave this resize iff needed to speed other operations
    current_frame = cv2.resize(current_frame, (constants.RESIZE_W, constants.RESIZE_H))
    current_frame = cv2.flip(current_frame, 1) # mirror

    # For each frame, detect landmarks 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=current_frame)
    detection_result = detector.detect(mp_image)

    # ------------------ Layer 2 (Input) ------------------
    # Compute low-level features (bounding rectangle + kinetic energy)
    bbox_left = None
    bbox_right = None
    velocities, landmarks, estimated = body_estimator.update(detection_result)
    ke = ke_processor.update(landmarks, velocities)
        
    bbox = get_bounding_rectangle(current_frame, landmarks)
    if bbox is not None:
        smooth_bbox = wall_butterworth_filter.filter([bbox[0], bbox[1]])
        bbox_left = int(smooth_bbox[0])
        bbox_right = int(smooth_bbox[1])
    
    #DEBUG
    if constants.DEBUG_KE:
        ke_history.append(ke*constants.MAX_KE) if ke is not None else ke_history.append(0.0)
        landmarks_history.append(estimated)

    # ------------------ Direct Mapping ------------------
    #TODO: use world landmarks for better accuracy for kinetic energy?
    is_ready = is_user_ready(current_frame, landmarks)
    current_state = mapping.update(detection_result.pose_landmarks, landmarks, ke, is_ready, bbox_left, bbox_right) 

    # ------------------ Layer 2 (Output) ------------------
    match current_state:
        case State.IDLE:
            output_frame = cv2.GaussianBlur(current_frame, (15, 15), 0)
            output_frame = overlay_logo(output_frame, logo, constants.RESIZE_W // 2, constants.RESIZE_H // 2)

        case State.PLAYING:
            # DEBUG
            import numpy as np
            if bbox is not None:
                bounding_rect = np.array([[bbox_left, bbox[3]], [bbox_left, bbox[2]], [bbox_right, bbox[2]], [bbox_right, bbox[3]]], dtype=np.int32)# DEBUG
                current_frame = draw_bounding_rectangle(current_frame, bounding_rect) 
            current_frame = draw_landmarks_on_image(current_frame, detection_result.pose_landmarks)

            output_frame = draw_walls(current_frame, bbox_left, bbox_right)
            output_frame, ke_display = draw_energy_bar(output_frame, ke, constants.THRESHOLD_KE, ke_display)
    
        case State.CLOSING:
            left, right = compute_wall_positions(mapping)
            output_frame = draw_walls(current_frame, left, right)

        case State.INTERRUPTION:
            output_frame = draw_message(current_frame, message="WARNING: Someone is disturbing the game!")

    # ------------------ Layer 1 (Output) ------------------

    #DEBUG
    if constants.DEBUG_KE:
        ke_graph_image = draw_cv_graph(ke_history, landmarks_history, estimated, output_frame.shape[1], output_frame.shape[0], 
                                       constants.MAX_KE, constants.FPS, constants.PLOT_WINDOW_SECONDS, y_label="Kinetic Energy", 
                                       threshold=constants.THRESHOLD_KE*constants.MAX_KE)
        
        output_frame = stack_images_horizontal([output_frame, ke_graph_image])

    cv2.imshow("Crushing Walls", output_frame)

    if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
        cap.release()
        break

# Closing video capture device
cv2.destroyAllWindows() 
cv2.waitKey(1)