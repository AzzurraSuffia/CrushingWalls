# General constants
FPS = 30
RESIZE_W = 640 # otherwise the screen is too little for visitors
RESIZE_H = 480
MAX_MISSED_FRAMES = 5

# Filters constants
WALL_CUTOFF = 2.0
WALL_ORDER = 2
VELOCITY_CUTOFF = 2.0
VELOCITY_ORDER = 2
KE_CUTOFF = 2.0
KE_ORDER = 2

# Kinetic energy computation
TOTAL_MASS = 80 # can mass be estimated somehow?
USE_ANTHROPOMETRIC_TABLES = True
APPLY_KE_FILTERING = True
MAX_KE = 150.0  # Adjust based on expected max kinetic energy
THRESHOLD_KE = 0.15 # Adjust based on expected max kinetic energy

# Mediapipe
MODEL_PATH="models\\pose_landmarker_lite.task"

# Graphics
LOGO_PATH="images\\logo.png"