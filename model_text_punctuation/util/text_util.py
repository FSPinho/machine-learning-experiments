import codecs
import json
import re
from os import path
from typing import List, Dict, Tuple

import pandas
from tqdm import tqdm

import config
# noinspection PyBroadException
from util.logging import Log


def load_words(lang="pt_br") -> Tuple[Dict[str, int], int]:
    words_dict = {
        **config.WORDS_CODES_CODES
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
    return text in list(config.WORDS_CODES_CODES.keys())


def print_tokens(cs, msg="", indent=1):
    def pad(txt, length=10):
        txt = "".join(str(txt)[0:length]).replace("\n", r"\n")
        return str(txt) + "".join([" "] * max(0, length - len(str(txt))))

    Log.i(msg + " | ".join(map(pad, cs)), indent=indent)


def split_tokens_from_code(code, tokens, verbose=False):
    if not isinstance(tokens, (list, tuple)):
        tokens = [tokens]

    tokens_tmp = []

    if verbose:
        progress = tqdm(total=len(tokens), ncols=84, desc=(" " * 15) + "Processing")
    else:
        progress = None

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


def load_text_tokens(source="simple", verbose=False) -> List[str]:
    with codecs.open(path.join(config.PHRASES_PATH, f"{source}.txt"), encoding="UTF-8") as file:
        tokens = file.read()
        tokens = re.sub(r"(\s)\s+", r"\1", tokens)
        p_codes = list(config.WORDS_CODES_CODES.keys())

        for p_code in p_codes:
            verbose and Log.i("Precessing code:", json.dumps(p_code), indent=2)
            tokens = split_tokens_from_code(p_code, tokens, verbose=verbose)
            verbose and Log.i(f"{len(tokens)} tokens found.", indent=3)

        return list(filter(lambda t: t != " ", tokens))


def tokens_to_dataset(tokens: List[str], words: Dict[str, int], target=".", force_equality=False):
    def get_code(item):
        if isinstance(item, int):
            return item
        elif item in words:
            return words[item]
        return config.WORDS_UNKNOWN_CODE

    dataset_t = []
    dataset_f = []
    dataset_t_raw = []
    dataset_f_raw = []

    progress = tqdm(total=len(tokens), ncols=84, desc=(" " * 15) + "Processing")

    for word_index, word in enumerate(tokens):
        progress.update(1)

        row = []
        has_target = False
        for i in range(word_index, word_index - config.PHRASES_LENGTH, -1):
            t = tokens[i] if i >= 0 else config.WORDS_EMPTY_CODE
            if t == target:
                has_target = True
            row.insert(0, config.WORDS_EMPTY_CODE if has_target else t)

        # row = sorted(row, key=lambda x: str(x))
        row_raw = row
        row = [get_code(t) for t in row]

        if config.WORDS_UNKNOWN_CODE not in row:
            if word_index < len(tokens) - 1 and tokens[word_index + 1] == target:
                row.append(1)
                dataset_t.append(row)
                dataset_t_raw.append(row_raw)
            else:
                row.append(0)
                dataset_f.append(row)
                dataset_f_raw.append(row_raw)

    progress.close()

    if force_equality:
        dataset = dataset_f[0: int(len(dataset_t) * 1)] + dataset_t
        dataset_raw = dataset_f_raw[0: int(len(dataset_t_raw) * 1)] + dataset_t_raw
    else:
        dataset = dataset_f + dataset_t
        dataset_raw = dataset_f_raw + dataset_t_raw

    return dataset, dataset_raw


def load(dataset):
    words, word_count = load_words()
    tokens = load_text_tokens(dataset, verbose=True)

    dataset, dataset_raw = tokens_to_dataset(tokens, words, target=".")
    dataset, dataset_raw = pandas.DataFrame(dataset), pandas.DataFrame(dataset_raw)

    x, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

    print(dataset)
    print(dataset_raw)

    return dataset_raw, x, y
