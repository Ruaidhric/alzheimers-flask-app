# Ruaidhrí
# Convert the keras model to tflite

import tensorflow as tf

model = tf.keras.models.load_model("Models/tensorflow_model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("Models/alzheimers_model.tflite", "wb") as f:
    f.write(tflite_model)
