from __future__ import division, print_function
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Load model
MODEL_PATH = 'resnet152V2_model.h5'
model = load_model(MODEL_PATH)

# Store results in memory (replace with DB in production)
uploaded_data = []

# Disease info
disease_info = {
    0: {
        "label": "Diseased Cotton Leaf",
        "disease": "Bacterial Blight",
        "treatment": "Use copper-based bactericides. Remove infected leaves."
    },
    1: {
        "label": "Diseased Cotton Plant",
        "disease": "Verticillium Wilt",
        "treatment": "Use crop rotation, resistant varieties, and improve drainage."
    },
    2: {
        "label": "Fresh Cotton Leaf",
        "disease": "No disease detected",
        "treatment": "No treatment needed. Continue regular monitoring."
    },
    3: {
        "label": "Fresh Cotton Plant",
        "disease": "No disease detected",
        "treatment": "No treatment needed. Maintain healthy practices."
    }
}

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    x = np.expand_dims(img_array, axis=0)

    prediction = model.predict(x)
    confidence = np.max(prediction)
    class_idx = int(np.argmax(prediction))

    # Confidence threshold for cotton detection
    if confidence < 0.80:
        return None  # Not cotton (or too uncertain)

    return disease_info.get(class_idx)



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', uploads=uploaded_data)

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    f = request.files['file']
    if f.filename == '':
        return redirect(url_for('index'))

    basepath = os.path.dirname(__file__)
    uploads_folder = os.path.join(basepath, 'static/uploads')
    os.makedirs(uploads_folder, exist_ok=True)

    unique_filename = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
    file_path = os.path.join(uploads_folder, unique_filename)
    f.save(file_path)

    result = model_predict(file_path)

    if result:
        uploaded_data.append({
            'filename': f'static/uploads/{unique_filename}',
            'label': result['label'],
            'disease': result['disease'],
            'treatment': result['treatment']
        })
    else:
        # If image is not confidently cotton
        uploaded_data.append({
            'filename': f'static/uploads/{unique_filename}',
            'label': "Invalid Image",
            'disease': "This is not a cotton plant or leaf.",
            'treatment': "Please upload a valid cotton plant or leaf image."
        })

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=5001, debug=True)
