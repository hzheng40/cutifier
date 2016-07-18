# CS 4475 PROJECT 2
# HONGRUI ZHENG
# CUTIFIER

import cv2
import sys
import numpy as np
import math

#1. import the training files

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier('ojoD.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')
#noseCascade = cv2.CascadeClassifier('nose.xml')

#2. start capturing video
vidCap = cv2.VideoCapture(0)

#3. load graphic files
eyeRaw = cv2.imread("eyes2.png")
eye = cv2.cvtColor(eyeRaw, cv2.COLOR_BGR2GRAY)
eyeHeight = eye.shape[0]
eyeWidth = eye.shape[1]

blushRaw = cv2.imread("blush.png")
blush = cv2.resize(blushRaw,None,fx=0.15, fy=0.15, interpolation = cv2.INTER_CUBIC)
blush = cv2.cvtColor(blush, cv2.COLOR_BGR2GRAY)
blushHeight = blush.shape[0]
blushWidth = blush.shape[1]

mustRaw = cv2.imread('must.jpg')
must = cv2.resize(mustRaw,None,fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
must = cv2.cvtColor(must, cv2.COLOR_BGR2GRAY)
mustHeight = must.shape[0]
mustWidth = must.shape[1]


#3. video loop, operation on each frame
while True:
	flag, frame = vidCap.read()
	#flag, frameOrig = vidCap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
		facegray = gray[y:y+h, x:x+w]
		facecolor = frame[y:y+h, x:x+w]
		eyes = eyeCascade.detectMultiScale(facegray, minNeighbors=5, scaleFactor=1.1, flags=cv2.cv.CV_HAAR_SCALE_IMAGE, minSize=(80,80))
		mouths = mouthCascade.detectMultiScale(facegray, minNeighbors=12, scaleFactor=1.1, flags=cv2.cv.CV_HAAR_SCALE_IMAGE, minSize=(40,45), maxSize=(120,120))
		#noses = noseCascade.detectMultiScale(facegray, minNeighbors=12, scaleFactor=1.1, flags=cv2.cv.CV_HAAR_SCALE_IMAGE, minSize=(10,10), maxSize=(90,60))
		
		# if eyes.any():	
		# 	(x1, y1, w1, h1) = eyes[0]
		# 	for r in range(0, h1):
		# 		for c in range (0, w1):
		# 				eyes[y1+r][x1+c] = facecolor[r][c]

		for br in range(0, blushHeight):
			for bc in range(0, blushWidth):
				if blush[br][bc] > 20:
					facecolor[220+br][70+bc][2] += 50
					facecolor[220+br][290+bc][2] += 50


		for (x1, y1, w1, h1) in eyes:



			#facecolor[y1:y1+w1][x1:x1+h1] = eye[0:w1][0:h1]
			if (y1 <= y+math.floor(h/2) - 100):
				for r in range(0, eyeHeight):
					for c in range(0, eyeWidth):
						if eye[r][c] < 10:
							continue
						else:
							facecolor[y1+math.floor(h1)-eyeHeight+r-10][x1+math.floor(w1)-eyeWidth+c-30] = eye[r][c]

				#cv2.rectangle(facecolor, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)

		for (x2, y2, w2, h2) in mouths:
			if (y2 >= y+math.floor(h/2) - 50):
				for mr in range(0, mustHeight):
					for mc in range(0, mustWidth):
						if must[mr][mc] < 100:
							facecolor[mr+200][mc+120] = must[mr][mc]
				#cv2.rectangle(facecolor, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)

		# nose detection deleted since too unstable
		#for (x3, y3, w3, h3) in noses:
		# 	cv2.rectangle(facecolor, (x3, y3), (x3+w3, y3+h3), (0, 0, 0), 2)

	cv2.imshow('vid', frame)



	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
vidCap.release()
cv2.destroyAllWindows()
