from enum import Enum, auto
from dataclasses import dataclass

@dataclass
class Counters:
    ready: int = 0
    energy: int = 0
    close: int = 0


@dataclass
class Thresholds:
    max_ready: int
    max_energy: int
    max_close: int
    energy_threshold: float


class State(Enum):
    """
    Enumeration of FSM states for the interactive wall installation.
    """
    IDLE = auto() 
    PLAYING = auto()
    CLOSING = auto()


class InteractionFSM:
    """
    Finite State Machine controlling the interactive wall installation.

    Manages transitions between IDLE, PLAYING, and CLOSING
    states based on user presence, movement energy, and tracking loss.
    """
    def __init__(self, max_ready, max_energy, max_close, energy_threshold):
        """
        Initialize the FSM with counters and thresholds.

        Args:
            max_ready (int): Number of frames required for user readiness to start PLAYING.
            max_energy (int): Number of frames user can stay below energy threshold before CLOSING.
            max_close (int): Number of frames the CLOSING animation lasts.
            threshold (float): Minimum energy level to avoid triggering CLOSING state.
        """
        self.state = State.IDLE

        self.counters = Counters()
        self.thresholds = Thresholds(max_ready, max_energy, max_close, energy_threshold)

        self.closing_bbox_left_start = None
        self.closing_bbox_right_start = None

    def update(self, pose_landmarks, estimated_landmarks, energy, starting_condition_met, bbox_left, bbox_right):
        """
        Update the FSM state based on current user input and system conditions.

        Args:
            pose_landmarks (list): Detected user landmarks in the current frame.
            estimated_landmarks (list): None if landmarks were not estimated.
            energy (float): Current user energy level.
            starting_condition_met (bool): True if the user meets the starting criteria to begin PLAYING.
            bbox_left (int): Left coordinate of the user's bounding rectangle.
            bbox_right (int): Right coordinate of the user's bounding rectangle.

        Returns:
            State: The current FSM state after evaluation and any transitions.
        """

        if self.state == State.PLAYING:

            # Landmark detection failed
            if not estimated_landmarks:
                self.counters.energy = 0
                self.state = State.IDLE

            # Low energy
            elif energy < self.thresholds.energy_threshold:
                self.counters.energy += 1

                if self.counters.energy >= self.thresholds.max_energy:
                    self.counters.energy = 0
                    self.state = State.CLOSING
                    self.closing_bbox_left_start = bbox_left
                    self.closing_bbox_right_start = bbox_right

            else:
                self.counters.energy = 0

        elif self.state == State.CLOSING:

            self.counters.close += 1
            if self.counters.close >= self.thresholds.max_close:
                self.counters.close = 0
                self.state = State.IDLE

        elif self.state == State.IDLE:

            if starting_condition_met:
                self.counters.ready += 1
                if self.counters.ready >= self.thresholds.max_ready:
                    self.counters.ready = 0
                    self.state = State.PLAYING
            else:
                self.counters.ready = 0

        return self.state