import pandas
from numpy.random import shuffle

import config
from util import text_util
from util.logging import Log
from util.model import define_model

LANG = "pt-br"
DATASET = "harry_potter_pt_br"

dataset, dataset_raw = text_util.load(dataset=DATASET)
shuffle(dataset)
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
model.fit([X] * 3, y, epochs=2, batch_size=16, workers=4)
model.save("model.h5")

loss, acc = model.evaluate([X] * 3, y, verbose=0)
Log.i(f"Model score = {'%.2f%%' % (acc * 100)}")
