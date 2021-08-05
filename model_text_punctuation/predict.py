from tensorflow.keras.models import load_model, Model

from util import text_util

raw, X, y = text_util.load("simple")

model: Model = load_model("model.h5")
predictions = model.predict([X] * 3, verbose=True)

print(raw)
print(predictions)

phrase = ""
for index, row in enumerate(raw.to_numpy()):
    prediction = predictions[index] > 0.5
    phrase += f"{' '.join([letter for letter in row if isinstance(letter, str)])}{'.' if prediction else ''}\n\n"
print(f"Prediction:\n {phrase}")
