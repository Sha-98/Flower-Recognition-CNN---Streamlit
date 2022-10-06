import streamlit as st
import tensorflow as tf 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('my_model.h5')
  return model
model=load_model()
st.write("""
          # Welcome to Flower Classification
          """)

file = st.file_uploader("Please upload a Flower Image" , type=["jpg","png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
  size = (180,180)
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img = np.asarray(image)
  img_reshape = img[np.newaxis,...]
  prediction = model.predicting(img_reshape)
  return prediction

if file in None:
  st.text("Please Upload an Image File")
else:
  image = Image.open(file)
  st.image(image, use_column_width=True)
  prediction = import_and_predict(image, model)
  class_names = ['Daisy', 'Dandelion', 'Rose' , 'Sunflower', 'Tulip']
  string = "The Flower in the Image is most likely is : "+class_name[np.argmax(predictions)]
  st.success(string)

