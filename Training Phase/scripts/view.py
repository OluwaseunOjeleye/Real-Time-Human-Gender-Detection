# ----------------------------------------------
# Annotation View
# ----------------------------------------------

import os
import cv2
import sys
import numpy as np

def view(MODE):
	path="dataset/1.jpg"

	target_image = cv2.imread(path)
	path=path.replace(".jpg",".txt")
	path=path.replace(".png",".txt")
	lines=open(path).readlines()
	
	c=0
	# data="0 0.162061692650334 0.449599596816976 0.06243429844098 0.101834482758621"
	# data=data.split(" ")
	# cls=c
	# x=int(float(data[1])*target_image.shape[1])
	# y=int(float(data[2])*target_image.shape[0])
	# w=int(float(data[3])*target_image.shape[1])
	# h=int(float(data[4])*target_image.shape[0])

	# color=(0,0,255)
	# thickness=3
	# cv2.rectangle(target_image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, thickness)
	# cv2.putText(target_image, str(cls), (x,y+16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
	# c=c+1


	for line in lines:
		data=line.split(" ")

		cls=int(data[0]) #c
		x=int(float(data[1])*target_image.shape[1])
		y=int(float(data[2])*target_image.shape[0])
		w=int(float(data[3])*target_image.shape[1])
		h=int(float(data[4])*target_image.shape[0])

		color=(0,0,255)
		thickness=3
		cv2.rectangle(target_image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, thickness)
		cv2.putText(target_image, str(cls), (x,y+16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250));
		c=c+1

		cv2.imshow("agegender", target_image)

		if cv2.waitKey(0) & 0xFF == ord('q'):
			cv2.destroyAllWindows()


	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()


def main(argv):
	MODE="vivahand"
	if len(sys.argv) == 2:
		MODE = sys.argv[1]
	else:
		print("usage: python annotation_view.py [vivahand/widerface/fddb]")
		sys.exit(1)
	if(MODE!="vivahand" and MODE!="widerface" and MODE!="fddb"):
		print("Unknown mode "+MODE)
		sys.exit(1)
	view(MODE)

if __name__=='__main__':
	main(sys.argv[1:])

