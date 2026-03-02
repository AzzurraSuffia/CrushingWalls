import copy
import time
import numpy as np

import config.constants as constants

class BodyEstimator:
    """
    Estimates body landmark velocities and predicts missing landmarks.
    """

    def __init__(self, alpha, max_missing_count=15, apply_filtering=False, velocity_filter=None):
        """
        Initialize the BodyEstimator.

        Args:
            alpha (float): Velocity decay factor for missing frames.
            max_missing_count (int): Max frames to predict missing landmarks.
            apply_filtering (bool): Apply optional velocity filtering.
            velocity_filter: Filter object with a `filter` method.
        """
        self.prev_velocities = None
        self.prev_time = None
        self.prev_landmarks = None

        self.applying_filtering = apply_filtering
        self.velocity_filter = velocity_filter

        self.missing_counter = 0
        self.max_missing_count = max_missing_count
        self.alpha = alpha

    def update(self, detection_result):
        """
        Update landmark velocities or predict missing ones.

        Args:
            detection_result: Object containing detected pose_landmarks.

        Returns:
            velocities (np.ndarray or None): Current landmark velocities.
            landmarks: Current or predicted landmarks.
            estimated (bool): True if landmarks were estimated.
        """
        curr_time = time.time()
        estimated = False

        # No detection or first frame
        if detection_result.pose_landmarks is None or len(detection_result.pose_landmarks) == 0:
            if self.prev_time is None or self.prev_landmarks is None or self.missing_counter >= self.max_missing_count:
                velocities, landmarks, estimated = None, None, False
            else:
                velocities, landmarks = self._predict_missing(curr_time)
                estimated = True

        else:
            # Use only the first detected person
            landmarks = detection_result.pose_landmarks[0]

            # First frame initialization
            if self.prev_landmarks is None:
                n_landmarks = len(landmarks)
                self.prev_landmarks = landmarks
                self.prev_velocities = np.zeros((n_landmarks, 3))
                self.prev_time = curr_time
                return self.prev_velocities, landmarks, estimated

            # Compute velocities
            velocities = np.array([
                self._first_order_derivative(
                    np.array([lm.x, lm.y, lm.z]),
                    np.array([prev_lm.x, prev_lm.y, prev_lm.z]),
                    curr_time, self.prev_time
                ) if lm.visibility > constants.VISIBILITY_THRESHOLD else np.zeros(3)
                for lm, prev_lm in zip(landmarks, self.prev_landmarks)
            ])

            # Optional filtering
            if self.applying_filtering and self.velocity_filter is not None:
                v_flat = velocities.reshape(-1)
                v_filtered = self.velocity_filter.filter(v_flat)
                velocities = v_filtered.reshape(len(landmarks), 3)

            self.missing_counter = 0

        # Update history only if valid
        if velocities is not None and landmarks is not None:
            self.prev_velocities = velocities
            self.prev_landmarks = landmarks
            self.prev_time = curr_time

        return velocities, landmarks, estimated
    
    @staticmethod
    def _first_order_derivative(curr_value, prev_value, curr_time, prev_time):
        """
        Compute first-order derivative (velocity) between two frames.

        Returns:
            np.ndarray: Velocity vector or None if invalid.
        """
        result = None
        if curr_value is not None and prev_value is not None and curr_time is not None and prev_time is not None:
            dt = curr_time - prev_time
            result = (curr_value - prev_value) / dt
        return result
    
    def _predict_missing(self, curr_time):
        """
        Predict landmarks and velocities for missing frames.

        Args:
            curr_time (float): Current timestamp.

        Returns:
            velocities (np.ndarray): Predicted velocities.
            landmarks: Predicted landmark positions.
        """
        dt = curr_time - self.prev_time

        # Apply velocity decay
        velocities = self.alpha * self.prev_velocities 
        landmarks = copy.deepcopy(self.prev_landmarks)

        # Derive landmarks from velocities
        for v, lm in zip(velocities, landmarks):
            lm.x += v[0] * dt
            lm.y += v[1] * dt
            lm.z += v[2] * dt

        self.missing_counter += 1
        
        return velocities, landmarks