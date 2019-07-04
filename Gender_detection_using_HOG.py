import cv2
import numpy as np

def computeHogForImage(image):
    hog = cv2.HOGDescriptor()
    resized_image = cv2.resize(image, (128, 128))
    hist = hog.compute(resized_image)
    hist1D = np.ravel(hist)
    return hist1D

def trainusingSVM(featuresList, labelsList, modelName):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    a = np.asarray(featuresList)
    b = np.array(labelsList)
    # print(a.shape)
    # print(b.shape)
    # print(a)
    # print(b)
    # print(labelsList)
    svm.train(a, cv2.ml.ROW_SAMPLE, b)
    svm.save(modelName)