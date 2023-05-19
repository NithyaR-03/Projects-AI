import os
import shutil
import zipfile
import random
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn import preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.keras.applications.xception import preprocess_input

model=load_model(r"animal3.h5",compile=False)
app=Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")
@app.route('/home')
def home():
    return render_template("index.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        img_data=preprocess_input(x)
        pred=np.argmax(model.predict(img_data),axis=1)
        index=['COVID','Lung_Opacity','Normal','Viral Pneumonia']
        text=" You Have Disease : " +str(index[pred[0]])
    return text

if __name__=='__main__':
    app.run()