import cv2
import os




output_loc=('D:\\College\\Compuer Vision\\Project\\Code final\\Genrated Frames\\man\\')
try:
 os.mkdir(output_loc)
except OSError:
  pass

# Start capturing the feed
cap = cv2.VideoCapture('1.mp4')
# Find the number of frames
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
print ("Number of frames: ", video_length)
count = 0
print ("Converting video..\n")
# Start converting the video
while cap.isOpened():
  # Extract the frame
  ret, frame = cap.read()
  # Write the results back to output location.
  cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
  count = count + 1
  # If there are no more frames left
  if (count > (video_length-1)):
                # Release the feed
                cap.release()
                # Print stats
                print ("Done extracting frames.\n%d frames extracted" % count)

                break
