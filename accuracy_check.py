import tensorflow as tf
from tensorflow import keras
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator

# Load the model
model = tf.keras.models.load_model('cat_dog_classifier.h5')

# Prepare the test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Evaluate the model
loss, accuracy = model.evaluate(test_set)
print(f'Accuracy: {accuracy * 100:.2f}%')