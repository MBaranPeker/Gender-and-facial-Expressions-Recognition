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

def Hog (Path):
    global acc
    global hap
    global ang,sad,nat,surp,fear,dis
    Path=path
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

    return hap,sad,nat,ang,surp,fear,dis


print('happy sad angry surprise natural,fear, disgust',hap,sad,ang,surp,nat,fear,dis)
print('Right prediction: ', acc)
print('False prediction: ', (24-acc))
print((acc/24)*100)
#####################################################################################################################



Face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
output_loc = ('D:\\College\\Compuer Vision\\Project\Code final\\Genrated Frames\\output\\')
path = 'D:\\College\\Compuer Vision\\Project\\Dataset Emotions\\train\\'
images = os.listdir(path)
framenumber=0
f4 = list()
HaarList = list()
## hof features extraction

def hof(frame1,frame2):
    Big = list()
    images = list()
    NewB = list()
    musk = np.zeros((8, 8))
    feature2 = list()
    feature3 = []

    for o in range(0, 8):
        for p in range(0, 8):
            x = frame2[8 * o:8 * o + 8, p * 8:p * 8 + 8]
            y = frame1[8 * o:8 * o + 8, p * 8:p * 8 + 8]

            prvs = cv.cvtColor(y, cv.COLOR_BGR2GRAY)
            hsv = np.zeros_like(y)
            hsv[..., 1] = 255

            next = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

            hist = cv2.calcHist([hsv[..., 0]], [0], None, [10], [0, 180])
            histv = cv2.calcHist([hsv[..., 2]], [0], None, [10], [0, 255])

            a = np.array(hist, np.float32)
            b = np.array(histv, np.float)


            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow('frame2', bgr)

            feature2 = np.concatenate((a, b), axis=0)
            feature2_array = np.array(feature2)
            x, y = np.shape(feature2_array)
            feature2 = np.reshape(feature2_array, (x * y))
            print('Y val',y)
            print('X val', x)
            feature3.append(feature2)
    return feature3

def HaarDetect(List):
    for j in (images):
        image = cv2.imread(path + j)

        Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Faces = Face_xml.detectMultiScale(Gray, 1.3, 5)

        for (x, y, w, h) in Faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (150, 60, 255), 2)

            GrayBorder = Gray[y:y + h, x:x + w]
            ColorBorder = image[y:y + h, x:x + w]
            cropped = image[y:y + h, x:x + w]

        image = cv2.resize(cropped, (64, 64))
        HaarList.append(image)

        try:
            os.mkdir(output_loc)
        except OSError:
            pass

    return HaarList

acc2 = 0
hap2 = 0
ang2 = 0
sad2 = 0
nat2 = 0
surp2 = 0
dis2 = 0
fear2 = 0
data_set_features2 = list()
counter2 = 1
image_hof_feature2=list()
def Hoffunction(Path):
        Path=path
        global acc2
        global hap2
        global ang2
        global sad2
        global nat2
        global surp2
        global dis2
        global fear2
        HaarList=HaarDetect(images)
        frame1 = HaarList[0]
        t = len(HaarList)
        Test2 = cv2.ml.SVM_load("hof.dat")

        for i in range(0, t):

            frame2 = HaarList[counter]
            image_hof_feature2 = hof(frame1, frame2)

            image_hof_feature2=np.array(image_hof_feature2)

            hi,wd = np.shape(image_hof_feature2)
            image_hof_feature=np.reshape(image_hof_feature2,(hi*wd))

            data_set_features2.append(image_hof_feature2)
            newData_set_feature = np.array(data_set_features2)

            TP2 = Test2.predict(newData_set_feature)
            for k in images:

                if 'happy' in k and TP2[1][0] == 0.:
                    acc2 += 1
                    hap2 += 1
                if 'sad' in k and TP2[1][0] == 1.:
                    acc2 += 1
                    sad2 += 1
                if 'angry' in k and TP2[1][0] == 2.:
                    acc2 += 1
                    ang2 += 1
                if 'surprise' in k and TP2[1][0] == 3.:
                    acc2 += 1
                    surp2 += 1
                if 'natural' in k and TP2[1][0] == 4.:
                    acc2 += 1
                    nat2 += 1
                if 'fear' in k and TP2[1][0] == 5.:
                    acc2 += 1
                    fear2 +=1
                if 'disgust' in k and TP2[1][0] == 6.:
                    acc2 +=1
                    dis2 +=1
                frame1 = frame2
                counter = counter + 1

                if counter == t:
                    break
        return hap2,sad2,ang2,surp2,nat2,fear2,dis2


print("image feature", image_hof_feature2.shape)
print("dataset feature", data_set_features2.shape)

print('happy sad angry surprise natural', hap2, sad2, ang2, surp2, nat2, fear2, dis2)
print('Right prediction: ', acc2)
print('False prediction: ', (25 - acc2))
print((acc2 / 25) * 100)