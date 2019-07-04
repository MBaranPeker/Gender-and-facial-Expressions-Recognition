import cv2
import os
import numpy as np

output_loc=('D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\input\\')
path=('D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\input\\')

try:
 os.mkdir(output_loc)
except OSError:
  pass
Video = cv2.VideoCapture('vid1.mp4')
images = os.listdir(path)
length=56
FrameRate=round(Video.get(cv2.CAP_PROP_FPS))
OutPutPath='D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\output\\'
outputcropped='D:\\College\\Compuer Vision\\Project\\14\\mult persond video\\cropped\\'
ImagesList=[]
Big=[]
EmotionTest = cv2.ml.SVM_load("emotion2.dat")
GenderTest=cv2.ml.SVM_load("gender.dat")
hog = cv2.HOGDescriptor()
count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
EmoLabel=''
GenderLabel=''
man=0
man2=0
woman=0
woman2=0
hap=0
hap2=0
sad=0
sad2=0
ang=0
ang2=0
nat=0
nat2=0
sur=0
sur2=0
fear=0
fear2=0
dis=0
dis2=0
def main(frame):
   global hap
   global sad
   global ang
   global dis
   global nat
   global sur
   global fear
   global Big
   global man
   global woman
   global count

   Face_xml=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   image=frame
   Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   Faces = Face_xml.detectMultiScale(Gray, 1.3, 5)

   for (x, y, w, h) in Faces:
       if x >300 :
           cv2.rectangle(image, (x, y), (x + w, y + h), (150, 60, 255), 2)
           Xlist=[]
           Ylist=[]
           # Xlist.appednd(x)
           # Ylist.append(y)
           GrayBorder = Gray[y:y + h, x: x + w]
           ColorBorder = image[y:y + h, x:x + w]
           cropped = image[y:y + h, x:x + w]
           resized_image = cv2.resize(cropped, (64, 128))

           H_hog = hog.compute(resized_image)

           Big.append(H_hog)
           BN = np.array(Big)
           x2, y2, z2 = np.shape(BN)
           NewB = np.reshape(BN, (x2, y2 * z2))
           print('list shape',NewB.shape)

           TP = EmotionTest.predict(NewB)
           TG= GenderTest.predict(NewB)
           if TP[1][0] == 0.:
               man += 1
           if TP[1][0] == 1.:
               woman += 1

           if TG[1][0] == 0.:
               hap+=1
           if TG[1][0] == 1.:
               sad+=1
           if TG[1][0] == 2.:
               ang+=1
           if TG[1][0] == 3.:
               sur+=1
           if TG[1][0] == 4.:
               nat+=1
           if TG[1][0] == 5.:
               fear+=1
           if TG[1][0] == 6.:
               dis+=1

           Big = []


           # cv2.putText(image[100:180, 250:380], EmoLabel, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
           # cv2.putText(image[100:180, 400:550], GenderLabel, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


           # while(1):
           #   cv2.imshow('img',image)
           #   k = cv2.waitKey(30) & 0xff
           #   if k == 27:
           #      break

           # cv2.imwrite(outputcropped + "/%#05d.jpg" % (count + 1), cropped)
           count = count + 1
           Big=[]


   return image,man,woman,x,y,hap,sad,ang,sur,nat,fear,dis


def maintwo(frame):
   global hap2
   global sad2
   global ang2
   global dis2
   global nat2
   global sur2
   global fear2
   global Big
   global man2
   global woman2
   global count

   man2 = 0
   woman2 = 0
   hap2 = 0
   sad2 = 0
   ang2 = 0
   nat2 = 0
   sur2 = 0
   fear2 = 0
   dis2 = 0

   Face_xml=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   image=frame
   Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   Faces = Face_xml.detectMultiScale(Gray, 1.3, 5)
   X=0
   Y=0

   for (x, y, w, h) in Faces:
       if x>600:
           break
       cv2.rectangle(image, (x, y), (x + w, y + h), (150, 60, 255), 2)
       Xlist=[]
       Ylist=[]
       # Xlist.appednd(x)
       # Ylist.append(y)
       GrayBorder = Gray[y:y + h, x: x + w]
       ColorBorder = image[y:y + h, x:x + w]
       cropped = image[y:y + h, x:x + w]
       resized_image = cv2.resize(cropped, (64, 128))

       H_hog = hog.compute(resized_image)

       Big.append(H_hog)
       BN = np.array(Big)
       x2, y2, z2 = np.shape(BN)
       NewB = np.reshape(BN, (x2, y2))
       print('list shape',NewB.shape)

       TP = EmotionTest.predict(NewB)
       TG= GenderTest.predict(NewB)
       if TG[1][0] == 0.:
           man2 += 1
       if TG[1][0] == 1.:
           woman2 += 1

       if TP[1][0] == 0.:
           hap2 += 1
       if TP[1][0] == 1.:
           sad2+= 1
       if TP[1][0] == 2.:
           ang2 += 1
       if TP[1][0] == 3.:
           sur2 += 1
       if TP[1][0] == 4.:
           nat2 += 1
       if TP[1][0] == 5.:
           fear2 += 1
       if TP[1][0] == 6.:
           dis2 += 1

       Big = []
       X=x
       Y=y

       # cv2.putText(image[100:180, 250:380], EmoLabel, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
       # cv2.putText(image[100:180, 400:550], GenderLabel, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


       # while(1):
       #   cv2.imshow('img',image)
       #   k = cv2.waitKey(30) & 0xff
       #   if k == 27:
       #      break

       # cv2.imwrite(outputcropped + "/%#05d.jpg" % (count + 1), cropped)
       count = count + 1
       Big=[]


   return image,man2,woman2,X,Y,hap2,sad2,ang2,sur2,nat2,fear2,dis2


def VidToFrames(outpath):
   output_loc=outpath
   video_length = int(Video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
   print("Number of frames: ", video_length)
   count = 0
   print("Converting video..\n")
   # Start converting the video
   while Video.isOpened():
       # Extract the frame
       ret, frame = Video.read()
       # Write the results back to output location.
       cv2.imwrite(output_loc + "/%#05d.jpg" % (count + 1), frame)
       count = count + 1
       # If there are no more frames left
       if (count > (video_length - 1)):
           # Release the feed
           Video.release()
           # Print stats
           print("Done extracting frames.\n%d frames extracted" % count)
           break




# def vid_to_frames(cap,NumOfDividedVideo):
NumOfDividedVideo=2
Interval=round(56/NumOfDividedVideo)
NumberOfFramesPerIntervals=round(Interval*FrameRate)
MidPoint =round(NumberOfFramesPerIntervals / 2)
StartFrame = round(MidPoint - 5)
EndFrame = round(MidPoint + 5)

# StartFrame=4
# EndFrame=14
count=0
for k in range(1,NumOfDividedVideo+1):

    for i, j in enumerate(images):
        if i in range(StartFrame,EndFrame):
            frame = cv2.imread(path + j)

            ImagesList.append(frame)
            cv2.imwrite(OutPutPath + "/%#05d.jpg" % (count + 1), frame)

            img1,man,woman,x,y,hap,sad,ang,sur,nat,fear,dis=main(frame)
            img2,man2,woman2,x2,y2,hap2,sad2,ang2,sur2,nat2,fear2,dis2=maintwo(frame)


            count =count +1

    if man <woman:
        GenderLabel = 'Man'
    else:
        GenderLabel = 'Woman'
    if man2 < woman2:
        GenderLabel2 = 'Man'
    else:
        GenderLabel2 = 'Woman'

    max1=max(hap,sad,ang,sur,nat,fear,dis)
    max2=max(hap2,sad2,ang2,sur2,nat2,fear2,dis2)

    if max1 == hap:
        EmoLabel1='Happy'
    if max1 == sad:
        EmoLabel1='Sad'
    if max1 == ang:
        EmoLabel1 = 'Angry'
    if max1 == sur:
        EmoLabel1 = 'Surprised'
    if max1 == nat:
        EmoLabel1 = 'Neutral'
    if max1 == fear:
        EmoLabel1 = 'Fear'
    if max1 == dis:
        EmoLabel1 = 'Disgust'

    if max2 == hap2:
        EmoLabel2 = 'Happy'
    if max2 == sad2:
        EmoLabel2 = 'Sad'
    if max2 == ang2:
        EmoLabel2 = 'Angry'
    if max2 == sur2:
        EmoLabel2 = 'Surprised'
    if max2 == nat2:
        EmoLabel2 = 'Neutral'
    if max2 == fear2:
        EmoLabel2 = 'Fear'
    if max2 == dis2:
        EmoLabel2 = 'Disgust'


        # cv2.putText(frame, EmoLabel, (x, y - 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # for m in Xlist:
    cv2.putText(frame, GenderLabel, (x + 150, y- 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, GenderLabel2, (x2 + 150, y2 - 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame,EmoLabel1 , (x, y - 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, EmoLabel2, (x2, y2 - 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('img', frame)

    cv2.waitKey(30)





    MidPoint=NumberOfFramesPerIntervals/2
    StartFrame=StartFrame+NumberOfFramesPerIntervals
    EndFrame=EndFrame+NumberOfFramesPerIntervals
