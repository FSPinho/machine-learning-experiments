from tensorflow.keras.models import load_model, Model

from util import text_util

raw, X, y = text_util.load("simple")

model: Model = load_model("model.h5")
predictions = model.predict([X] * 3, verbose=True)

for index, row in enumerate(raw.to_numpy()):
    word = " ".join([letter for letter in row if isinstance(letter, str)])
    prediction = predictions[index] > 0.5

    print(f"{word}{'. [PERIOD]' if prediction else ''}")
