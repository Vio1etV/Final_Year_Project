from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import tensorflow_addons as tfa
from PIL import Image

app = Flask(__name__)

custom_objects = {"F1Score": tfa.metrics.F1Score}

model = load_model('C:/Users/MSI-PC/Desktop/Level 6 curriculum/大四下'
                   '/Final Year Project/Z. ProgressBackup'
                   '/Model Training Results/ModelH5File/Best H5/Version8_run5-DIR(Latest Best).h5',
                   custom_objects=custom_objects)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/Recognition', methods=['GET', 'POST'])
def recognition():
    if request.method == 'GET':
        # Render the HTML page where the user can upload an image
        return render_template('Recognition.html')

    elif request.method == 'POST':
        # Handle the image file uploaded via POST request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Read the image file directly from the uploaded file
            image = Image.open(file.stream)
            # Ensure the image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((224, 224))  # Resize the image to match model's input requirements
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image array

            # Predict using the loaded model
            prediction_probabilities = model.predict(img_array)
            benign_prob = float(prediction_probabilities[0][0])
            malignant_prob = float(prediction_probabilities[0][1])
            predicted_class = 'benign' if benign_prob > malignant_prob else 'malignant'

            return jsonify({
                'prediction': predicted_class,
                'probabilities': {
                    'benign': benign_prob,
                    'malignant': malignant_prob
                }
            })
        else:
            return jsonify({'error': 'Method not allowed'}), 405


if __name__ == '__main__':
    app.run(debug=True)
