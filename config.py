MODEL_ARCH = 'deepsleep'
EVENT_CHANNEL = 'EDF Annotations'

## LOAD EDF FILEPATH AND THEN TRANSFORM TO NUMPY
BASE_PATH = '/media/pc/UUI/sleep/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette'
OUTPUT_DIR = 'data_2018/'
## MIGHT WANT TO CHANGE IT WITH A LOCAL DIRECTORY

SLEEP_STAGES_DICT = {'Sleep stage W':5, 'Sleep stage 1':3, 'Sleep stage 2':2, 'Sleep stage 3':1,
            'Sleep stage 4':0, 'Sleep stage R':4, 'Movement time':6}

EPOCHS = 5
BATCH_SIZE = 16
EPOCH_SEC_SIZE = 30
SAMPLE_RATE = 44100  # Hertz
DURATION = 5  # Seconds
SELECT_CH = "EEG Fpz-Cz"
INPUT_SHAPE = (3000,1)
FS = 100
N_CLASSES = 5



# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}
