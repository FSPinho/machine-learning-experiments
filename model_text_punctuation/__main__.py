import pandas

from util import text_util
from util.logging import Log
from util.measure import Measure

LANG = "pt-br"

with Measure(f"Load {LANG} words"):
    words = text_util.load_words()
    Log.i(f"{len(words)} words loaded.", indent=2)

print("\n")

with Measure("Load sample text tokens"):
    tokens = text_util.load_text_tokens()
    Log.i(f"{len(tokens)} tokens loaded.", indent=2)

print("\n")

with Measure("Load dataset from tokens"):
    dataset = pandas.DataFrame(text_util.tokens_to_dataset(tokens, words, verbose=True))
    Log.i(f"{len(dataset)} rows loaded.", indent=2)

print("\n")

print(dataset)
