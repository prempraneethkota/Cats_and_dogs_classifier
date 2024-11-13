from flask import Flask, request, render_template
import tensorflow as tf
import io
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('cat_dog_classifier.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape(1, 64, 64, 3)
        img_array = img_array / 255.0
        prediction = model.predict(img_array)
        class_label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
        return f'This image is classified as a {class_label}.'
    return 'Invalid file'

if __name__ == '__main__':
    app.run(debug=True)
