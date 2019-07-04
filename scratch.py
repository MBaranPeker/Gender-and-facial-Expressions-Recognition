import cv2
import cv2 as cv
import numpy as np
import os
Face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


output_loc=('F:\\#year 4\\1st semester\\computer vision \\Code final\\Genrated Frames\\output')


L = list()
Big = list()
images = list()
HaarList=list()
NewB = list()
musk = np.zeros((8, 8))
feature1=list()
feature2=list()
feature3=list()
path = 'F:\\#year 4\\1st semester\\computer vision \\Milestone1 vision\\data set videos and images\\all\\MPI_large_centralcam_hi_cawm_complete\\train\\'
images = os.listdir(path)
framenumber=0
for j in images:
    image = cv2.imread(path+j)

    Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Faces = Face_xml.detectMultiScale(Gray, 1.3, 5)

    for (x, y, w, h) in Faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (150, 60, 255), 2)

        GrayBorder = Gray[y:y + h, x:x + w]
        ColorBorder = image[y:y + h, x:x + w]
        cropped = image[y:y + h, x:x + w]


    if 'happy' in j:
        L.append(0)
    if'sad' in j:
        L.append(1)
    if 'angry' in j:
        L.append(2)
    if'surprise' in j:
        L.append(3)
    if 'natural' in j:
        L.append(4)
    image=cv2.resize(cropped,(64,64))
    HaarList.append(image)

    try:
     os.mkdir(output_loc)
    except OSError:
      pass
    cv2.imwrite(output_loc + "/%#05d.happy.jpg" % (framenumber+1), image)
    framenumber=+1
counter=1
musk2=np.ones((8,8))
musk1=np.ones((8,8))

frame1=HaarList[0]
t=len(HaarList)
for i in HaarList:
   frame2=HaarList[counter]
   for o in range(0,8):
      for p in range (0,8):
           x=frame2[8*o:8*o+7,p*8:p*8+7]
           y=frame1[8*o:8*o+7,p*8:p*8+7]

           prvs = cv.cvtColor(y,cv.COLOR_BGR2GRAY)
           hsv = np.zeros_like(y)
           hsv[...,1] = 255

           next = cv.cvtColor(x,cv.COLOR_BGR2GRAY)
           flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

           mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
           hsv[...,0] = ang*180/np.pi/2
           hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
           # for o in range(0,8):
           #    for p in range (0,8):
           #         x=frame2[8*o:8*o+8,p*8:p*8+8]
           #         y=frame1[o:o+8,p:p+8]
           hist = cv2.calcHist([hsv[...,0]],[0],None,[10],[0,255])
           histv =cv2.calcHist([hsv[...,2]],[0],None,[10],[0,255])
          # np.concatenate(hist,histv)
          #  iii=cv.hconcat(hist,histv)
          #  iiii=cv.vconcat(hist,feature1)
          #  i2=cv.vconcat(histv,feature2)
          #  f1=np.concatenate(hist[1],histv[1])

           # for a in range(0,9):
           #feature1.append(hist[a])
           #feature2.append(histv[a])
           feature2=hist[p], histv[p]
           feature3.append(feature2)
           #feature2.append(histv[0])
           #ii=np.concatenate(feature1.index(0),feature2.index(0))
           #f1=np.concatenate(feature1.index(),feature1.index())
           # feature2.append(f1)

   bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
   cv.imshow('frame2',bgr)
   # k = cv.waitKey(30) & 0xff
   # if k == 27:
   #   break
   # elif k == ord('s'):
   #   cv.imwrite('opticalfb.png',frame2)
   #   cv.imwrite('opticalhsv.png',bgr)


   frame1=frame2
   counter=counter+1
   if counter== t:
       break
# cv.destroyAllWindows()



svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

BN = np.array(feature3, np.float32)
LN = np.array(L, np.int)
svm.train(BN, cv2.ml.ROW_SAMPLE, LN)

svm.save('hof.dat')
