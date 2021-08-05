from os import path

BASE_DIR = path.dirname(__file__)

DATASETS_PATH = path.join(BASE_DIR, "datasets")

WORDS_PATH = path.join(DATASETS_PATH, "words")
WORDS_EMPTY_CODE = 0
WORDS_UNKNOWN_CODE = -1
WORDS_CODES = {
    " ": 0,
    "\n": 1,
    "...": 2,
    ".": 3,
    ",": 4,
    ":": 5,
    ";": 6,
    "(": 7,
    ")": 8,
    "?": 9,
    "!": 10,
    "@": 11,
}

PHRASES_PATH = path.join(DATASETS_PATH, "samples")
PHRASES_LENGTH = 16
