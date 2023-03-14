import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('hotdog.h5')
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

st.set_page_config(layout = 'centered')

st.title("HotDog Detection APP :hotdog:")
st.subheader("About the Model")
st.write("""InceptionV3 is a deep convolutional neural network architecture used for image recognition tasks. It was developed by Google and is based on the concept of "inception modules," which are small neural network modules that are stacked together to form a larger network. InceptionV3 has achieved high accuracy on various image classification tasks,
 and it is often used as a pre-trained model for transfer
learning in computer vision applications.InceptionV3 has 48 layers and consists of multiple branches that allow the network to capture different levels of image features. The network is trained on the ImageNet dataset, which contains over a million images across 1,000 different classes. InceptionV3 uses techniques such as batch normalization, dropout, and regularization to prevent overfitting and improve generalization performance. The architecture has been shown to outperform previous state-of-the-art models on several benchmarks,
including the ImageNet Large Scale Visual Recognition Challenge.""")

st.subheader("How to Use")
st.write("1.Upload an image")
st.write("2.The image can be anything")
st.write("3.It'll tell if it's a HotDog are not :)")



image = st.file_uploader('Upload a Image file', type = ['jpg','png','jpeg'])
if image is None:
    st.text('Upload an Image file')

else:
    image = Image.open(image)
    st.image(image)
    img = image.resize((300,300))
    x = img_to_array(img)
    x /= 255.0
    x = np.expand_dims(x, axis=0)
    predict = model.predict(x)

    st.header("HotDog" if predict < 0.5 else "Not Hotdog")
    st.balloons()
    