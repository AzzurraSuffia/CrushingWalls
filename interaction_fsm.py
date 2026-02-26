from enum import Enum, auto


class State(Enum):
    # auto(): automatically generates increasing integer values to be assignt o states
    IDLE = auto() 
    PLAYING = auto()
    CLOSING = auto()
    INTERRUPTION = auto()


class InteractionFSM:

    def __init__(self, max_count, max_close, threshold):
        self.state = State.IDLE

        self.MAX = max_count
        self.MAX_CLOSE = max_close
        self.threshold = threshold

        self.ready_counter = 0
        self.energy_counter = 0
        self.disturb_counter = 0
        self.absence_counter = 0
        self.close_counter = 0

        self.closing_bbox_left_start = None
        self.closing_bbox_right_start = None

    def update(self, pose_landmarks, energy, starting_condition_met, bbox_left, bbox_right):
        """
        Update FSM state based on current inputs.
        Returns the current state.
        """

        if self.state == State.PLAYING:

            if len(pose_landmarks) > 1:
                self.disturb_counter += 1
                self.energy_counter = 0
                self.absence_counter = 0

                if self.disturb_counter >= self.MAX:
                    self.disturb_counter = 0
                    self.state = State.INTERRUPTION

            elif not pose_landmarks:
                self.absence_counter += 1
                self.energy_counter = 0
                self.disturb_counter = 0

                if self.absence_counter >= self.MAX:
                    self.absence_counter = 0
                    self.state = State.IDLE

            elif energy < self.threshold:
                self.energy_counter += 1
                self.disturb_counter = 0
                self.absence_counter = 0

                if self.energy_counter >= self.MAX:
                    self.energy_counter = 0
                    self.state = State.CLOSING
                    self.closing_bbox_left_start = bbox_left
                    self.closing_bbox_right_start = bbox_right

            else:
                self._reset_activity_counters()

        elif self.state == State.CLOSING:

            self.close_counter += 1
            if self.close_counter >= self.MAX_CLOSE:
                self.close_counter = 0
                self.state = State.IDLE

        elif self.state in (State.IDLE, State.INTERRUPTION):

            if starting_condition_met:
                self.ready_counter += 1
                if self.ready_counter >= self.MAX:
                    self.ready_counter = 0
                    self.state = State.PLAYING
            else:
                self.ready_counter = 0

        return self.state

    def _reset_activity_counters(self):
        self.disturb_counter = 0
        self.energy_counter = 0
        self.absence_counter = 0