import pandas

import config
from util import text_util
from util.logging import Log
from util.measure import Measure
from util.model import define_model

LANG = "pt-br"
DATASET = "harry_potter_pt_br"

with Measure(f"Load {LANG} words"):
    words, word_count = text_util.load_words()
    Log.i(f"{len(words)} ({word_count}) words loaded.", indent=2)

print("\n")

with Measure("Load sample text tokens"):
    cache_path = f"/tmp/{DATASET}"

    # noinspection PyBroadException
    try:
        tokens = pandas.read_pickle(cache_path).tolist()
        Log.i(f"{len(tokens)} CACHE tokens loaded.", indent=2)
    except Exception:
        tokens = text_util.load_text_tokens(DATASET, verbose=True)
        pandas.Series(tokens).to_pickle(cache_path)
        Log.i(f"{len(tokens)} tokens loaded.", indent=2)

print("\n")

with Measure("Load dataset from tokens"):
    dataset, dataset_raw = text_util.tokens_to_dataset(tokens, words, target=".", force_equality=True)
    dataset, dataset_raw = pandas.DataFrame(dataset), pandas.DataFrame(dataset_raw)

    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
    counts = y.value_counts()

    Log.i(f"{len(dataset)} rows loaded.", indent=2)
    Log.i(f"{counts[1]} TRUE occurrences.", indent=2)
    Log.i(f"{counts[0]} FALSE occurrences.", indent=2)

print("\n\n")

print(dataset_raw)
print(dataset)

print("\n\n")
Log.i("Testing models...")
print("\n")

model = define_model(config.PHRASES_LENGTH, word_count)
model.fit([X] * 3, y, epochs=10, batch_size=16, workers=4)
model.save("model.h5")

loss, acc = model.evaluate([X] * 3, y, verbose=0)
Log.i(f"Model score = {'%.2f%%' % (acc * 100)}")

#
#
#
#
#
#
#
#
#
#
#
#
if False:
    models = [
        ("Baseline", DummyClassifier()),
        ("MLPClassifier", MLPClassifier(
            random_state=1, shuffle=True, hidden_layer_sizes=(16,),
            learning_rate_init=0.125, max_iter=2000,
            activation="logistic", learning_rate="adaptive"
        )),
        ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=1)),
    ]

    for model_name, model in models:
        with Measure(f"{model_name}"):
            scores = cross_val_score(model, X, y, scoring="accuracy", cv=10, n_jobs=-1, verbose=False)
            # Log.i(f"Scores = {','.join(map(lambda s: '%.2f%%' % (s * 100), scores))}", indent=2)
            Log.i(f"Mean Score = {'%.2f%%' % (mean(scores) * 100)}", indent=2)

        print("\n")
