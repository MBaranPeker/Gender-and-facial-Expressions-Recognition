import numpy as np
import cv2
import os
Face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
output='D:\\College\\Compuer Vision\\Project\\Dataset\\new\\'
path = 'D:\\College\\Compuer Vision\\Project\\Dataset\\manwomandetection\\train\\TrainHahahaha\\'
pathtany = ''
images = os.listdir(path)
for j in images:
    Img = cv2.imread(path+j)


    Gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    Faces = Face_xml.detectMultiScale(Gray, 1.3, 5)

    for (x, y, w, h) in Faces:
        cv2.rectangle(Img, (x, y), (x + w, y + h), (150, 60, 255), 2)

        GrayBorder = Gray[y:y + h, x:x + w]
        ColorBorder = Img[y:y + h, x:x + w]
        cropped=Img[y:y + h, x:x + w]
        cv2.imwrite(cropped,output)




