import codecs
import re
from os import path
from typing import List, Dict, Tuple

import pandas
from tqdm import tqdm

import config
# noinspection PyBroadException
from util.logging import Log


def load_words(lang=None) -> Tuple[Dict[str, int], int]:
    if not lang:
        lang = "pt_br"

    words_dict = {
        **config.WORDS_CODES
    }
    start_index = len(words_dict)

    with codecs.open(path.join(config.WORDS_PATH, f"{lang}.txt"), encoding="UTF-8") as file:
        words_raw = file.read().split("\n")
        words_raw = map(lambda w: w.strip().lower(), words_raw)
        words_raw = filter(lambda w: len(w), words_raw)
        words_raw = sorted(words_raw)

        for index, word in enumerate(words_raw):
            words_dict[word] = start_index
            start_index += 1

    return words_dict, start_index


def is_punctuation_code(text):
    return text in list(config.WORDS_CODES.keys())


def pad(txt, length=10):
    txt = "".join(str(txt)[0:length]).replace("\n", r"\n")
    return str(txt) + "".join([" "] * max(0, length - len(str(txt))))


def print_tokens(cs, msg="", indent=1):
    Log.i(msg + " | ".join(map(pad, cs)), indent=indent)


def split_tokens_from_code(code, tokens):
    if not isinstance(tokens, (list, tuple)):
        tokens = [tokens]

    tokens_tmp = []
    progress = tqdm(total=len(tokens), ncols=128, desc=f"Processing {pad(code, 6)}\t")

    for chunk_index, chunk in enumerate(tokens):
        progress and progress.update(1)

        if is_punctuation_code(chunk):
            tokens_tmp.append(chunk)
        else:
            chunk_parts = chunk.split(code)
            chunk_parts = list(map(lambda p: p.lstrip().rstrip(), chunk_parts))

            if len(chunk_parts) == 1:
                tokens_tmp.append(chunk)
            else:
                tmp = []

                for part_index, part in enumerate(chunk_parts):
                    tmp.append(part)

                    if part_index < len(chunk_parts) - 1:
                        tmp.append(code)

                tmp = map(lambda p: p if is_punctuation_code(p) else p.lstrip().rstrip().lower(), tmp)
                tmp = list(filter(lambda p: p or is_punctuation_code(p), tmp))

                tokens_tmp += tmp

    progress and progress.close()

    return tokens_tmp


def load_text_tokens(dataset=None, text=None, words=None) -> List[str]:
    if dataset:
        with codecs.open(path.join(config.PHRASES_PATH, f"{dataset}.txt"), encoding="UTF-8") as file:
            tokens = file.read()
    else:
        tokens = text or ""

    if not words:
        words = load_words()

    tokens = tokens[:100]
    tokens = "".join(filter(lambda t: t in words, list(tokens)))
    tokens = re.sub(r"(\s)\s+", r"\1", tokens)

    p_codes = list(config.WORDS_CODES.keys())

    for p_code in p_codes:
        tokens = split_tokens_from_code(p_code, tokens)

    return list(filter(lambda t: t != " ", tokens))


def tokens_to_dataset(tokens: List[str], words: Dict[str, int], target=".", force_equality=False):
    def get_code(item):
        if isinstance(item, int):
            return item
        elif item in words:
            return words[item]
        return config.WORDS_UNKNOWN_CODE

    return [[]], []


def load(dataset=None, text=None, lang=None):
    words, word_count = load_words(lang)
    tokens = load_text_tokens(dataset=dataset, text=text, words=words)
    dataset, dataset_raw = tokens_to_dataset(tokens, words, target=".")
    dataset, dataset_raw = pandas.DataFrame(dataset), pandas.DataFrame(dataset_raw)

    x, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

    return dataset, dataset_raw, x, y
