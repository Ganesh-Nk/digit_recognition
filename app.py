from flask import Flask, request, jsonify, render_template
import tensorflow as tf
#from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('model/my_cnn_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (28, 28))
    # img = img.reshape(1, 28, 28,1) / 255.0
    #prediction = model.predict(img)
    return img


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        data_url = request.json['image']
        img_data = base64.b64decode(data_url.split(',')[1])
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img = np.array(img)

    img = preprocess_image(img)
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({'digit': int(digit), 'confidence': confidence})


if __name__ == "__main__":
    app.run(debug=True)

