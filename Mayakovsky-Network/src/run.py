import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from train import text
from utils import generate_text

model = tf.keras.models.load_model('mayakovsky_trained.keras')

generated_text = generate_text(model, text, 100, 1.0)

print(generated_text)