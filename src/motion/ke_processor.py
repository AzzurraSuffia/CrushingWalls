import numpy as np

import config.constants as constants
import motion.masses as masses

class KEProcessor:
    """Processes landmark velocities to compute total kinetic energy."""

    def __init__(self, velocity_filter):
        """
        Initialize the kinetic energy processor.

        Args:
            velocity_filter: Optional filter object to smooth velocities.
        """

        self.velocity_filter = velocity_filter

        self.prev_detection = None
        self.prev_time = None

    def update(self, landmarks, velocities):
        """
        Compute the current kinetic energy from landmarks and velocities.

        Args:
            landmarks (list): List of current landmark positions.
            velocities (np.ndarray): Velocity vectors of the landmarks.

        Returns:
            float: Total kinetic energy. Returns 0.0 if input is invalid.
        """
         
        # Mass vector
        if constants.USE_ANTHROPOMETRIC_TABLES:
            masses_vector = masses.create_mass_vector(constants.TOTAL_MASS)
        else:
            masses_vector = None

        # Compute KE
        ke = self._compute_kinetic_energy(landmarks, velocities, masses_vector)

        if ke is None:
            ke = 0.0

        return ke
    
    def _compute_kinetic_energy(self, landmarks, velocities, masses=None):
        """
        Compute kinetic energy given landmarks, velocities, and masses.

        Args:
            landmarks (list): Landmark positions.
            velocities (np.ndarray): Landmark velocity vectors.
            masses (np.ndarray, optional): Mass of each landmark.

        Returns:
            float: Total kinetic energy, or None if inputs are invalid.
        """
         
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