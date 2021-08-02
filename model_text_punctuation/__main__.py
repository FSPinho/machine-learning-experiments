import pandas
from numpy import mean
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from util import text_util
from util.logging import Log
from util.measure import Measure

LANG = "pt-br"
# DATASET = "harry_potter_001_pt_br"
# DATASET = "harry_potter_pt_br"
DATASET = "the_lord_of_the_rings"
# DATASET = "simple"

with Measure(f"Load {LANG} words"):
    words = text_util.load_words()
    Log.i(f"{len(words)} words loaded.", indent=2)

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
    dataset, dataset_raw = text_util.tokens_to_dataset(tokens, words, target=".")
    dataset, dataset_raw = pandas.DataFrame(dataset), pandas.DataFrame(dataset_raw)

    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
    counts = y.value_counts()

    Log.i(f"{len(dataset)} rows loaded.", indent=2)
    Log.i(f"{counts[1]} TRUE occurrences.", indent=2)
    Log.i(f"{counts[0]} FALSE occurrences.", indent=2)

print(dataset_raw)

print("\n")

with Measure("Scale dataset"):
    X = pandas.DataFrame(StandardScaler().fit_transform(X, y))

print("\n\n")
Log.i("Testing models...")
print("\n")

models = [
    ("Baseline", DummyClassifier()),
    ("LogisticRegression", LogisticRegression(random_state=1)),
    ("Perceptron", Perceptron(random_state=1)),
    ("MLPClassifier", MLPClassifier(
        random_state=1, shuffle=True, hidden_layer_sizes=(16,),
        learning_rate_init=0.125, max_iter=2000, alpha=0.0000001,
        activation="logistic", learning_rate="constant"
    )),
    ("MLPClassifier", MLPClassifier(
        random_state=1, shuffle=True, hidden_layer_sizes=(16,),
        learning_rate_init=0.125, max_iter=2000, alpha=0.000001,
        activation="logistic", learning_rate="invscaling"
    )),
    ("MLPClassifier", MLPClassifier(
        random_state=1, shuffle=True, hidden_layer_sizes=(16,),
        learning_rate_init=0.125, max_iter=2000, alpha=0.00001,
        activation="logistic", learning_rate="adaptive"
    )),
    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=1)),
    # ("SVN", SVC(random_state=1)),
    # ("KNeighborsClassifier", KNeighborsClassifier()),
]

for model_name, model in models:
    with Measure(f"{model_name}"):
        scores = cross_val_score(model, X, y, scoring="accuracy", cv=10, n_jobs=-1, verbose=False)
        # Log.i(f"Scores = {','.join(map(lambda s: '%.2f%%' % (s * 100), scores))}", indent=2)
        Log.i(f"Mean Score = {'%.2f%%' % (mean(scores) * 100)}", indent=2)

    print("\n")
