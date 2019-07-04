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
path = 'D:\\College\\Compuer Vision\\Project\\Dataset\\manwomandetection\\owntest\\'
images = os.listdir(path)
#Test = pickle.load(open("SVModel.sav", "rb"))
Test = cv2.ml.SVM_load("gender.dat")
acc=0
man=0
woman=0
for i, j in enumerate(images):


        image = cv2.imread(path+j)

        Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Faces = Face_xml.detectMultiScale(Gray, 1.3, 5)

        for (x, y, w, h) in Faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (150, 60, 255), 2)

            GrayBorder = Gray[y:y + h, x:x + w]
            ColorBorder = image[y:y + h, x:x + w]
            cropped=image[y:y + h, x:x + w]

            cv2.imshow('img',cropped)

            resized_image = cv2.resize(cropped, (64, 128))


            h = hog.compute(resized_image)
            Big.append(h)

            BN = np.array(Big)
            x, y, z = np.shape(BN)
            NewB = np.reshape(BN, (x, y * z))
            print('list shape',NewB.shape)
            TP = Test.predict(NewB)

            #print(TP[1][0])
            if 'man' in j and TP[1][0] == 0.:
                acc+=1
                man+=1
            if 'bnat' in j and TP[1][0] == 1.:
                acc+=1
                woman+=1

            Big=[]
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break



# print('newb AFter',NewB.shape)
# print('bn before', BN.shape)

print('man and woman',man,woman)
print('Right prediction: ', acc)
print('False prediction: ', (10-acc))

print((acc/10)*100)
#TP1 = np.array(TP)
#BN = np.array(Big, np.float32)
#LN = np.array(L, np.int)
#print("Accuracy:", metrics.accuracy_score(LN, TP))

