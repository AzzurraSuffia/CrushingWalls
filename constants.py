# General constants
FPS = 30
RESIZE_W = 640 # otherwise the screen is too little for visitors
RESIZE_H = 480
MAX_MISSED_FRAMES = 5

# Filters constants
WALL_CUTOFF = 2.0
WALL_ORDER = 2
VELOCITY_CUTOFF = 3.0
VELOCITY_ORDER = 2

# Kinetic energy computation
TOTAL_MASS = 80 # can mass be estimated somehow?
USE_ANTHROPOMETRIC_TABLES = True
APPLY_KE_FILTERING = True
MAX_KE = 1000.0  # Adjust based on expected max kinetic energy
THRESHOLD_KE = 0.1 # Adjust based on expected max kinetic energy

# Mediapipe
MODEL_PATH="models\\pose_landmarker_lite.task"

# Graphics
LOGO_PATH="images\\logo.png"

# Mapping 
MAX_COUNT = FPS
MAX_CLOSE_SECONDS = 1
MAX_CLOSE = int(MAX_CLOSE_SECONDS * FPS) 
CLOSED_PAUSE = 5

# Velocity Decay
ALPHA = 0.9

# Debug energy
DEBUG_KE = True
PLOT_WINDOW_SECONDS = 5

# Center region
CENTER_X_MIN = 0.40
CENTER_X_MAX = 0.60
CENTER_Y_MIN = 0.30
CENTER_Y_MAX = 0.70
VISIBILITY_THRESHOLD = 0.5