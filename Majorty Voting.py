import cv2
import os
import numpy as np


Face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

L = list()
Big = list()
images = list()
NewB = list()
PathList=list()
hog = cv2.HOGDescriptor()
path1 = 'D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\interval 1\\'
path2 = 'D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\interval 2\\'
path3 = 'D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\interval 3\\'
path4 = 'D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\interval 4\\'
path5 = 'D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\intervaltwo 1\\'
path6 = 'D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\intervaltwo 2\\'
path7 = 'D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\intervaltwo 3\\'
path8 = 'D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\intervaltwo 4\\'
PathList.append(path1)
PathList.append(path2)
PathList.append(path3)
PathList.append(path4)
PathList.append(path5)
PathList.append(path6)
PathList.append(path7)
PathList.append(path8)


TruePredict=0
FalsePredict=0
Test = cv2.ml.SVM_load("gender.dat")
acc=0
man=0
woman=0
for s,k in enumerate(PathList):
    images = os.listdir(k)
    for i, j in enumerate(images):


            image = cv2.imread(k+j)

            Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            Faces = Face_xml.detectMultiScale(Gray, 1.3, 5)

            for (x, y, w, h) in Faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (150, 60, 255), 2)

                GrayBorder = Gray[y:y + h, x:x + w]
                ColorBorder = image[y:y + h, x:x + w]
                cropped=image[y:y + h, x:x + w]

                # cv2.imshow('img',cropped)

                resized_image = cv2.resize(cropped, (64, 128))


                h = hog.compute(resized_image)
                Big.append(h)

                BN = np.array(Big)
                x, y, z = np.shape(BN)
                NewB = np.reshape(BN, (x, y * z))
                # print('list shape',NewB.shape)
                TP = Test.predict(NewB)

                #print(TP[1][0])
                if TP[1][0] == 0.:
                    man+=1
                if TP[1][0] == 1.:
                    woman+=1
                Big=[]

            if man<woman:
                Label=woman
            else:
                Label=man

            if s<4 and Label== man:
                TruePredict+=1
            if s<4 and Label== woman:
                FalsePredict+=1
            if s>=4 and Label== man:
                FalsePredict+=1
            if s>=4 and Label== woman:
                TruePredict+=1


acc=(FalsePredict/(TruePredict+FalsePredict))


print('True Prediction= ',TruePredict)
print('False Prediction= ',FalsePredict)
print('Accurecy: ', acc*100)

# print('False prediction: ', (10-acc))

# print((acc/10)*100)

