# General constants
FPS = 30
VIDEO_PATH = "videos\\"
VIDEO_NAME = "offline_demo.mp4"
LIVE_INPUT = False
RESIZE_W = 640
RESIZE_H = 480

# Debug constants
DEBUG = True
PLOT_WINDOW_SECONDS = 5

# Filters constants
WALL_CUTOFF = 2.0
WALL_ORDER = 2
VELOCITY_CUTOFF = 3.0
VELOCITY_ORDER = 2

# Kinetic energy computation
TOTAL_MASS = 60
USE_ANTHROPOMETRIC_TABLES = True
APPLY_KE_FILTERING = True
MAX_KE = 100.0 
THRESHOLD_KE = 0.08*MAX_KE 

# Mediapipe
MODEL_PATH="models\\pose_landmarker_lite.task"

# Graphics
LOGO_PATH="images\\logo.png"

# Mapping 
MAX_READY = FPS
MAX_ENERGY = 2 * FPS
MAX_CLOSE = FPS 
CLOSED_PAUSE = 5

# Velocity Decay
ALPHA = 0.9

# Center region
CENTER_X_MIN = 0.40
CENTER_X_MAX = 0.60
CENTER_Y_MIN = 0.30
CENTER_Y_MAX = 0.70
VISIBILITY_THRESHOLD = 0.5