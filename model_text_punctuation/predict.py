from tensorflow.keras.models import load_model, Model

from util import text_util

raw, X, y = text_util.load("simple")

model: Model = load_model("model.h5")
predictions = model.predict([X] * 3, verbose=True)

phrase = ""
for index, row in enumerate(raw.to_numpy()):
    if isinstance(row[-1], str):
        prediction = predictions[index] > 0.5
        phrase += f"{row[-1]}{'.' if prediction else ''} "
print(f"Prediction:\n {phrase}")
