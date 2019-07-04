import cv2
import cv2 as cv
import numpy as np
import os
Face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

output_loc = ('D:\\College\\Compuer Vision\\Project\Code final\\Genrated Frames\\output\\')

path = 'D:\\College\\Compuer Vision\\Project\\Dataset Emotions\\train\\'
framenumber=0
L = list()
f4 = list()
HaarList = list()

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
            feature3.append(feature2)
    return feature3


images = os.listdir(path)

for s, j in enumerate(images):
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

    cv2.imwrite(output_loc + "/%#05d.happy.jpg" % (framenumber + 1), image)
    framenumber = +1

    if s==len(images)-1:
        break
    if 'happy' in j:
        L.append(0)
    if 'sad' in j:
        L.append(1)
    if 'angry' in j:
        L.append(2)
    if 'surprise' in j:
        L.append(3)
    if 'natural' in j:
        L.append(4)
    if 'fear' in j:
        L.append(5)
    if 'disgust' in j:
        L.append(6)
counter = 1
frame1 = HaarList[0]
t = len(HaarList)
data_set_features = []

for i in range(0, t):
    frame2 = HaarList[counter]
    image_hof_feature = hof(frame1, frame2)
    hi,wd = len(image_hof_feature),len(image_hof_feature[0])
    image_hof_feature=np.array(image_hof_feature).reshape((hi*wd))

    data_set_features.append(image_hof_feature)
    print(np.array(data_set_features).shape)

    frame1 = frame2
    counter = counter + 1

    if counter == t:
        break
# cv.destroyAllWindows()
#print(feature3.shape)
data_set_features=np.array(data_set_features)
print("hiii",data_set_features.shape)


svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

BN = np.array(data_set_features, np.float32)
LN = np.array(L, np.int)
print(LN.shape)
svm.train(BN, cv2.ml.ROW_SAMPLE, LN)  ##################################################### eh al error da  !

svm.save('hof.dat')
