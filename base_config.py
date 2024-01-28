# Base configuration file - rename to "config.py"

# API
TOKEN = 'MAPILLARY-API-TOKEN'
ASYNC = True

# Coords
MIN_LAT = 1.237
MAX_LAT = 1.473
MIN_LNG = 103.605
MAX_LNG = 104.042
BBOX_SIZE = 0.001
BBOX_NUM_IMG = 30
COORD_ACC = 3

# Files
IMG_DIR = 'img'
LABELS_FILE = 'img/labels.csv'
MODEL_DIR = 'models'

# Data
IMG_SIZE = 576 # shortest edge
BATCH_SIZE = 32
SHUFFLE = True
PIN_MEMORY = True
DROP_LAST = True
