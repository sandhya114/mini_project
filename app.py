from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the saved models
verification_model = load_model('resnet501.h5')
classification_model = load_model('nclassify_model60.h5')

def preprocess_image(img_path, target_size=(256, 256)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']

        # Save the uploaded file
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        # Resize the image to (256, 256)
        img = Image.open(filepath)
        img = img.resize((256, 256))
        img.save(filepath)

        # Preprocess the image
        img_array = preprocess_image(filepath)

        # Verify if the image is a potato leaf
        verification_prediction = verification_model.predict(img_array)[0][0]
        print(f"Verification prediction: {verification_prediction}")
        is_potato_leaf = verification_prediction > 0.7  # Adjust threshold if necessary

        if is_potato_leaf:
            # Classify the potato leaf disease
            result = classification_model.predict(img_array)
            prediction = np.argmax(result)
            if prediction == 0:
                class_name = "Late Blight"
            elif prediction == 1:
                class_name = "Healthy"
            else:
                class_name = "Early Blight"
        else:
            class_name = "Not a potato leaf"

        return render_template('result.html', prediction=class_name, image_filename=filename),render_template('base.html', prediction=class_name)

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/disease_info/<disease>')
def disease_info(disease):
    # session['disease'] = disease  # Store the disease in session
    return render_template('disease_info.html', disease=disease)

@app.route('/contact/<disease>')
def contact(disease):
    # session['disease'] = disease  # Store the disease in session
    return render_template('contact.html', disease=disease)
    

if __name__ == '__main__':
    app.run(debug=True)
