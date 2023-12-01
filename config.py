import datetime

# Video Path
VIDEO_CONFIG = {
	"VIDEO_CAP":0,       # here we can put integer 0 for real time camera survillance ####
	"IS_CAM" : False,
	"CAM_APPROX_FPS": 3,
	"HIGH_CAM": False,
	"START_TIME": datetime.datetime(2020, 11, 5, 0, 0, 0, 0)
}

# Load YOLOv3-tiny weights and config
YOLO_CONFIG = {                                                       ####
	"WEIGHTS_PATH" : "YOLO/yolov4-tiny.weights",
	"CONFIG_PATH" : "YOLO/yolov4-tiny.cfg"
}

SHOW_PROCESSING_OUTPUT = True    #for video processing output       ####
SHOW_DETECT = True       #greem bounding box around peoplwe detected
DATA_RECORD = True       # for recording the processed_data generate from video    ####
DATA_RECORD_RATE = 5     # Data record rate (data record per frame)


RE_CHECK = False   # Restricted entry time (H:M:S)
RE_START_TIME = datetime.time(0,0,0)  # video start time 
RE_END_TIME = datetime.time(0,0,15)   # video end time

SD_CHECK = False
# Show violation count
SHOW_VIOLATION_COUNT = False
# Show tracking id
SHOW_TRACKING_ID = False
# Threshold for distance violation
SOCIAL_DISTANCE = 50

ABNORMAL_CHECK = True       # Check for abnormal crowd activity
ABNORMAL_MIN_PEOPLE = 2 # Min number of people to check for abnormal
ABNORMAL_ENERGY = 1866     
ABNORMAL_THRESH = 0.66

MIN_CONF = 0.3             # Threshold for human detection minumun confindence ....used in tracking.py
NMS_THRESH = 0.2           # Threshold for Non-maxima surpression

FRAME_SIZE = 1080          # Resize frame for processing                            ####

TRACK_MAX_AGE = 3          # Tracker max missing age before removing (seconds)      ####