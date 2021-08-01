import codecs
import json
import re
from os import path
from typing import List, Dict

import config
# noinspection PyBroadException
from util.logging import Log


def load_words(lang="pt_br") -> Dict[str, int]:
    words_dict = {}

    with codecs.open(path.join(config.WORDS_PATH, f"{lang}.txt"), encoding="UTF-8") as file:
        words_raw = file.read().split("\n")
        words_raw = map(lambda w: w.strip().lower(), words_raw)
        words_raw = filter(lambda w: len(w), words_raw)
        words_raw = sorted(words_raw)

        for index, word in enumerate(words_raw):
            words_dict[word] = index

    return words_dict


def is_punctuation_code(text):
    return text in list(config.WORDS_PUNCTUATIONS_CODES.keys())


def print_tokens(cs, msg="", indent=1):
    def pad(txt, length=16):
        txt = "".join(str(txt)[0:length])
        return str(txt) + "".join([" "] * max(0, length - len(str(txt))))

    Log.i(msg + " | ".join(map(pad, cs)), indent=indent)


def split_tokens_from_code(code, tokens, verbose=False):
    if not isinstance(tokens, (list, tuple)):
        tokens = [tokens]

    while True:
        did_split = False
        verbose and print_tokens(tokens)

        for chunk_index, chunk in enumerate(tokens.copy()):
            if is_punctuation_code(chunk):
                continue

            chunk_parts = chunk.split(code)
            chunk_parts = list(map(lambda p: p.lstrip().rstrip(), chunk_parts))

            if len(chunk_parts) > 1:
                verbose and print_tokens(chunk_parts, "Chunk parts ", indent=2)

                tmp = sum([[part, code] for part in chunk_parts], [])[:-1]
                tmp = map(lambda p: p if is_punctuation_code(p) else p.lstrip().rstrip().lower(), tmp)
                tmp = list(filter(lambda p: p or is_punctuation_code(p), tmp))

                verbose and print_tokens(tmp, "Chunk tmp ", indent=2)

                tokens = tokens[:chunk_index] + tmp + tokens[chunk_index + 1:]

                did_split = True
                break

        if not did_split:
            break

        verbose and print_tokens(tokens)

    verbose and print_tokens(tokens)

    return tokens


def load_text_tokens(source="simple", verbose=False) -> List[str]:
    with codecs.open(path.join(config.PHRASES_PATH, f"{source}.txt"), encoding="UTF-8") as file:
        tokens = file.read()
        tokens = re.sub(r"(\s)\s+", r"\1", tokens)

        p_codes = list(config.WORDS_PUNCTUATIONS_CODES.keys())
        for p_code in p_codes:
            verbose and Log.i("Code:", json.dumps(p_code))
            tokens = split_tokens_from_code(p_code, tokens, verbose=verbose)

        return tokens


def tokens_to_dataset(tokens: List[str], words: Dict[str, int], target=".", verbose=False):
    def get_code(item):
        if item in config.WORDS_PUNCTUATIONS_CODES:
            return config.WORDS_PUNCTUATIONS_CODES[item]
        elif item in words:
            return words[item]
        return config.WORDS_UNKNOWN_CODE

    dataset = []

    for word_index, word in enumerate(tokens):
        row = [config.WORDS_EMPTY_CODE] * max(0, config.PHRASES_LENGTH - word_index)
        row += tokens[max(0, word_index - config.PHRASES_LENGTH): word_index]

        if word == target:
            row += [1]
        else:
            row += [0]

        verbose and print_tokens(row, "Row: ", indent=2)

        dataset.append([get_code(i) for i in row[:-1]] + row[-1:])

    return dataset
