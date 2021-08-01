from os import path

BASE_DIR = path.dirname(__file__)

DATASETS_PATH = path.join(BASE_DIR, "datasets")

WORDS_PATH = path.join(DATASETS_PATH, "words")
WORDS_EMPTY_CODE = -1
WORDS_UNKNOWN_CODE = -2
WORDS_PUNCTUATIONS_CODES = {
    "\n": -3,
    "...": -4,
    ".": -5,
    ",": -6,
    ":": -7,
    ";": -8,
    "-": -9,
    "(": -10,
    ")": -11,
    "?": -12,
    "!": -13,
    "@": -14,
    " ": -15,
}

PHRASES_PATH = path.join(DATASETS_PATH, "samples")
PHRASES_LENGTH = 8
