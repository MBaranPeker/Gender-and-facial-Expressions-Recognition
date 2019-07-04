import cv2
import os
import numpy

output_loc=('D:\\College\\Compuer Vision\\Project\\Code final\\Genrated Frames\\output\\')
musk1=cv2.imread('D:\\College\\Compuer Vision\\Project\\Code final\\Genrated Frames\\output\\00002.happy.jpg')
musk1=cv2.resize(musk1,(8,8))
cv2.imwrite(output_loc + "/%#05d.happy.jpg" ,musk1)

