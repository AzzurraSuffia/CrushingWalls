import time
import numpy as np

import constants
import masses

class KE_Processor:

    def __init__(self, velocity_filter, ke_filter):

        self.velocity_filter = velocity_filter
        self.ke_filter = ke_filter

        self.prev_detection = None
        self.prev_time = None

    def update(self, detection_result):

        curr_time = time.time()

        if self.prev_time is None:
            self.prev_time = curr_time
            self.prev_detection = detection_result
            return 0

        # Mass vector
        if constants.USE_ANTHROPOMETRIC_TABLES:
            masses_vector = masses.create_mass_vector(
                constants.TOTAL_MASS
            )
        else:
            masses_vector = None

        # Compute KE
        ke = self._compute_kinetic_energy(
            detection_result,
            self.prev_detection,
            self.prev_time,
            curr_time,
            masses_vector,
            constants.APPLY_KE_FILTERING,
            self.velocity_filter
        )

        # Update references
        self.prev_detection = detection_result
        self.prev_time = curr_time

        # Normalize
        if ke is None:
            ke = 0
        else:
            ke = ke / constants.MAX_KE

        # Final filtering
        ke = self.ke_filter.filter(ke)

        return ke
    
    def _compute_kinetic_energy(self, current_detection_result, previous_detection_result, 
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
            self._first_order_derivative(np.array([curr_lm.x, curr_lm.y, curr_lm.z]),
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
    
    @staticmethod
    def _first_order_derivative(curr_value, prev_value, curr_time, prev_time):
        result = None
        if curr_value is not None and prev_value is not None and curr_time is not None and prev_time is not None:
            dt = curr_time - prev_time
            result = (curr_value - prev_value) / dt
        return result