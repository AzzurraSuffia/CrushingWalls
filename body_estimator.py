import copy
import time
import numpy as np

class BodyEstimator:
    def __init__(self, alpha, max_missing_count=15, apply_filtering=False, velocity_filter=None):
        self.prev_velocities = None
        self.prev_time = None
        self.prev_landmarks = None

        self.applying_filtering = apply_filtering
        self.velocity_filter = velocity_filter

        self.missing_counter = 0
        self.max_missing_count = max_missing_count
        self.alpha = alpha

    def update(self, detection_result):

        curr_time = time.time()
        
        if detection_result.pose_landmarks is not None and len(detection_result.pose_landmarks) > 0:

            # Use only the first detected person
            landmarks = detection_result.pose_landmarks[0]
            n_landmarks = len(landmarks)

            if self.prev_landmarks is None:
                self.prev_landmarks = landmarks
                self.prev_time = curr_time
                self.prev_velocities = np.zeros((len(landmarks), 3))
                return self.prev_velocities, landmarks

            velocities = np.array([
                        self._first_order_derivative(np.array([curr_lm.x, curr_lm.y, curr_lm.z]),
                        np.array([prev_lm.x, prev_lm.y, prev_lm.z]),
                        curr_time, self.prev_time)
            for curr_lm, prev_lm in zip(landmarks, self.prev_landmarks)
            ])

            # Optional filtering
            if self.applying_filtering and self.velocity_filter is not None:
                # Flatten velocities for filtering: (n_points * 3,)
                v_flat = velocities.reshape(-1)
                # Filter one "sample" per channel (vectorized)
                v_f = self.velocity_filter.filter(v_flat)
                velocities = v_f.reshape(n_landmarks, 3)

            self.missing_counter = 0
            
        else:
            if self.missing_counter < self.max_missing_count:
                velocities = None
                landmarks = None
            else:
                dt = curr_time - self.prev_time
                velocities = self.alpha * self.prev_velocities

                landmarks = copy.deepcopy(self.prev_landmarks)

                for v, lm in zip(velocities, landmarks):
                    lm.x += v[0] * dt
                    lm.y += v[1] * dt
                    lm.z += v[2] * dt

                self.missing_counter += 1

        self.prev_velocities = velocities
        self.prev_landmarks = landmarks
        self.prev_time = curr_time
        return velocities, landmarks
    
    @staticmethod
    def _first_order_derivative(curr_value, prev_value, curr_time, prev_time):
        result = None
        if curr_value is not None and prev_value is not None and curr_time is not None and prev_time is not None:
            dt = curr_time - prev_time
            result = (curr_value - prev_value) / dt
        return result