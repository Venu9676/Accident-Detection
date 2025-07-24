from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import ttk
from tkinter import filedialog
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import os
import cv2
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


main = tkinter.Tk()
main.title('AI-POWERED ACCIDENT DETECTION: TRAFFIC SURVEILLANCE IMAGES USING CNN WITH VGG16')
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")


global filename
global X, Y
global model
global categories,model_folder


model_folder = "model"

label = ['Accident','no Accident']

def uploadDataset():
    global filename,categories,categories,path
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")
    
def imageProcessing():
    text.delete('1.0', END)
    global X, Y, model_folder, filename,X_file,Y_file
    path = r"Dataset"
    model_folder = "model"
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    


    X_file = os.path.join(model_folder, "X.txt.npy")
    Y_file = os.path.join(model_folder, "Y.txt.npy")
    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
        print("X and Y arrays loaded successfully.")
    else:
        X = [] # input array
        Y = [] # output array
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(f'Loading category: {dirs}')
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img_array = cv2.imread(root+"/"+directory[j])
                    img_resized = resize(img_array, (128, 128, 3))
                    # Append the input image array to X
                    X.append(img_resized.flatten())
                    # Append the index of the category in categories list to Y
                    Y.append(categories.index(name))
        X = np.array(X)
        Y = np.array(Y)
        np.save(X_file, X)
        np.save(Y_file, Y)

    text.insert(END, "Dataset Preprocessing completed\n")

def Train_Test_split():
    global X,Y,x_train,x_test,y_train,y_test
    
    
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)
    
    text.insert(END,"Total samples found in training dataset: "+str(x_train.shape)+"\n")
    text.insert(END,"Total samples found in testing dataset: "+str(x_test.shape)+"\n")


def calculateMetrics(algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="Pastel1" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       

def Existing_ETC():
    global x_train,x_test,y_train,y_test,model_folder
    text.delete('1.0', END)
    
    model_filename = os.path.join(model_folder, "ETC_model.pkl")
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = ExtraTreesClassifier(n_estimators=10,max_depth=3)
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        print(f'Model saved to {model_filename}')

    y_pred = mlmodel.predict(x_test)
    calculateMetrics("Existing Extra Trees Classifier", y_pred, y_test)


def cnnModel():
    global X, Y, x_train, x_test, y_train, y_test, model_folder, categories, model, history, base_model,lr_scheduler,opt
    text.delete('1.0', END)
    
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X = X[indices]
        Y = Y[indices]
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X = X[indices]
        Y = Y[indices]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    
    x_train = x_train.reshape((-1, 128, 128, 3))  # VGG16 requires 224x224 input size
    x_test = x_test.reshape((-1, 128, 128, 3))
    y_train = to_categorical(y_train, num_classes=len(categories))  
    y_test = to_categorical(y_test, num_classes=len(categories))  
    
    Model_file = os.path.join(model_folder, "VGG16_model.json")
    Model_weights = os.path.join(model_folder, "VGG16_model_weights.h5")
    Model_history = os.path.join(model_folder, "history.pckl")
    num_classes = len(categories)

    if os.path.exists(Model_file):
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()    
        model.load_weights(Model_weights)
        model._make_predict_function()   
        print(model.summary())
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)
            acc = history['accuracy']
    else:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

        for layer in base_model.layers:
            layer.trainable = False  # Freeze pretrained layers

        model = Sequential([
            base_model,
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        
        hist = model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_test, y_test), shuffle=True, verbose=2,callbacks=[lr_scheduler])
        model.save_weights(Model_weights)            
        model_json = model.to_json()
        with open(Model_file, "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)
        with open(Model_history, 'rb') as f:
            accuracy = pickle.load(f)
            acc = accuracy['accuracy']
            acc = acc[9] * 100
            print("VGG16 Model Prediction Accuracy = "+str(acc))

    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1) 
    calculateMetrics("VGG16 Model", Y_pred_classes, y_test)



def graph():
    global history

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training & validation accuracy
    axs[0].plot(history['accuracy'])
    axs[0].plot(history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def predict():
    global model
    categories= ['Accident','no Accident']
    
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (128,128))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(-1,128,128,3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test/255
    
    X_test_features = model.predict(test)
    predict = np.argmax(X_test_features)
    img = cv2.imread(filename)
    img = cv2.resize(img, (500,500))
    cv2.putText(img, 'Classified as : '+categories[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Classified as : '+categories[predict], img)
    cv2.waitKey(0)
  

def close():
    main.destroy()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='AI-POWERED ACCIDENT DETECTION: TRAFFIC SURVEILLANCE IMAGES USING CNN WITH VGG16')
title.config(bg='SlateBlue1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Image Processing", command=imageProcessing)
processButton.place(x=20,y=150)
processButton.config(font=ff)

mlpButton = Button(main, text="Dataset Splitting", command=Train_Test_split)
mlpButton.place(x=20,y=200)
mlpButton.config(font=ff)

mlpButton = Button(main, text="Train Extra Trees Classifier", command=Existing_ETC)
mlpButton.place(x=20,y=250)
mlpButton.config(font=ff)


modelButton = Button(main, text="Train VGG16 Model", command=cnnModel)
modelButton.place(x=20,y=300)
modelButton.config(font=ff)

graphButton = Button(main, text="Accuracy & Loss", command=graph)
graphButton.place(x=20,y=350)
graphButton.config(font=ff)



predictButton = Button(main, text="Prediction from Test Image", command=predict)
predictButton.place(x=20,y=400)
predictButton.config(font=ff)



exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=450)
exitButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config(bg = 'azure')
main.mainloop()
