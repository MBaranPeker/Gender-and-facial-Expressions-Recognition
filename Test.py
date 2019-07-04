import cv2
import time
import cv2 as cv
import numpy as np
import pickle
import os
from sklearn import svm
from sklearn import metrics


Face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

L = list()
Big = list()
images = list()
NewB = list()
hog = cv2.HOGDescriptor()
path = 'D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\try\\'
images = os.listdir(path)
#Test = pickle.load(open("SVModel.sav", "rb"))
Test = cv2.ml.SVM_load("emotions.dat")
acc=0
hap=0
ang=0
sad=0
nat=0
surp=0
fear=0
dis=0
for i, j in enumerate(images):

    image = cv2.imread(path+j)

    Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Faces = Face_xml.detectMultiScale(Gray, 1.3, 5)

    for (x, y, w, h) in Faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (150, 60, 255), 2)

        GrayBorder = Gray[y:y + h, x:x + w]
        ColorBorder = image[y:y + h, x:x + w]
        cropped = image[y:y + h, x:x + w]

        cv2.imshow('img', cropped)

        resized_image = cv2.resize(cropped, (64, 128))

        h = hog.compute(resized_image)
        Big.append(h)

        BN = np.array(Big)
        x, y, z = np.shape(BN)
        NewB = np.reshape(BN, (x, y * z))
        print('list shape',NewB.shape)

        TP = Test.predict(NewB)

        #print(TP[1][0])
        if 'happy' in j and TP[1][0] == 0.:
            acc+=1
            hap+=1
        if 'sad' in j and TP[1][0] == 1.:
            acc+=1
            sad+=1
        if 'angry' in j and TP[1][0] == 2.:
            acc+=1
            ang+=1
        if 'surprise' in j and TP[1][0] == 3.:
            acc+=1
            surp+=1
        if 'natural' in j and TP[1][0] == 4.:
            acc+=1
            nat+=1
        if 'fear' in j and TP[1][0] == 5.:
            acc += 1
            fear += 1
        if 'disgust' in j and TP[1][0] == 6.:
            acc += 1
            dis += 1
        Big=[]




# print('newb AFter',NewB.shape)
# print('bn before', BN.shape)

print('happy sad angry surprise natural,fear, disgust',hap,sad,ang,surp,nat,fear,dis)
print('Right prediction: ', acc)
print('False prediction: ', (24-acc))

print((acc/24)*100)
#TP1 = np.array(TP)
#BN = np.array(Big, np.float32)
#LN = np.array(L, np.int)
#print("Accuracy:", metrics.accuracy_score(LN, TP))

