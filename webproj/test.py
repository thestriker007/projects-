import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("cataract_classifier_model_new.keras")

IMG_HEIGHT = 150
IMG_WIDTH = 150

img_path = "processed_images/test/cataract/image_304.png"

img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("normal Detected")
else:
    print("cataract Eye")