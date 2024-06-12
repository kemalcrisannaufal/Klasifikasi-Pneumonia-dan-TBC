from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)

categories = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

model = tf.keras.models.load_model('model88.h5')

def median_filtering(img):
    img = cv2.medianBlur(img, 3)
    return img

def contrast_stretching(img):
    in_min = np.percentile(img, 5)
    in_max = np.percentile(img, 95)
    out_min, out_max = img.min(), img.max()
    img_contrast = (img - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
    img_contrast = np.clip(img_contrast, out_min, out_max)
    return img_contrast

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255
    i = median_filtering(i)
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = contrast_stretching(i)
    i = np.expand_dims(i, axis=0)
    p = model.predict(i)
    p = np.argmax(p, axis=1)
    return categories[p[0]]

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['image']
        img_path = f'static/{img.filename}'
        img.save(img_path)
        p = predict_label(img_path)
    return render_template('index.html', prediction=p, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
