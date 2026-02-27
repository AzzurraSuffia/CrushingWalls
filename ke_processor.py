import numpy as np

import constants
import masses

class KE_Processor:

    def __init__(self, velocity_filter, ke_filter):

        self.velocity_filter = velocity_filter
        self.ke_filter = ke_filter

        self.prev_detection = None
        self.prev_time = None

    def update(self, landmarks, velocities):

        # Mass vector
        if constants.USE_ANTHROPOMETRIC_TABLES:
            masses_vector = masses.create_mass_vector(
                constants.TOTAL_MASS
            )
        else:
            masses_vector = None

        # Compute KE
        ke = self._compute_kinetic_energy(landmarks, velocities, masses_vector)

        # Normalize
        if ke is None:
            ke = 0.0
        else:
            ke = ke / constants.MAX_KE

        # Final filtering
        ke = self.ke_filter.filter(ke)

        return ke
    
    def _compute_kinetic_energy(self, landmarks, velocities, masses=None):

        # Ensure landmarks exist
        if landmarks is None or velocities is None:
            return None

        n_landmarks = len(landmarks)
        if masses is None:
            masses = np.ones(n_landmarks)

        # Compute total kinetic energy
        speed_squared = np.sum(velocities**2, axis=1)
        total_ke = 0.5 * np.sum(masses * speed_squared)

        return total_ke