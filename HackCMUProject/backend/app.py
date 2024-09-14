from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from flask_cors import CORS
from PIL import Image
import cv2  # Add this import

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('cancerDetector.keras')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("no files")
        return jsonify({'error': 'BRUDDA NO FILE'}), 400

    file = request.files['file']
    if file.filename == '':
        print("there's no file crodie")
        return jsonify({'error': 'SELECT A FILE BRUDA'}), 400

    img = Image.open(io.BytesIO(file.read())).convert('L')
    img = np.array(img)

    img = cv2.resize(img, (512, 512))

    img = img / 255.0

    img_array = img.reshape(1, 512, 512, 1)

    predictions = model.predict(img_array)
    probability = predictions[0][0]

    threshold = 0.3
    binary_result = 1 if probability >= threshold else 0

    print(f"Probability: {probability}, Binary result: {binary_result}")
    return jsonify({'predictions': binary_result})

if __name__ == '__main__':
    app.run(debug=True)
