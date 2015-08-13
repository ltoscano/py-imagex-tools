#!/usr/bin/python
# License: Creative Commons Zero http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
title           :OpenCVQuickFilters.py
description     :Capture image patch in real time from the camera 
					and plot auto correlation error surface to understand and measure 
					the stability of a patch
#usage          :python ImagePatchStabilityAnalysis.py
python_version  :2.7.6
depends         :numpy, opencv, matplotlib
author          :Yagnesh Revar
"""

import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

pState = "IDLE"
selectionPts = [ (-1,-1) for i in range(2) ] # 1st is constant, 2nd point keeps varying
gray = [] # realtime grayscale camera frame  
gray_bkp = []

def onMouseEvent(event, x, y, flags, param):
	# grab references to the global variables
	global selectionPts, pState, gray_bkp, gray

	# Update 2nd point during the intermediate state 
	if pState == "SELECTION_STARTED" or pState == "SELECTING":
		selectionPts[1] = (x, y)
		pState = "SELECTING"

	# if the left mouse button was clicked, record the selection starting
	if event == cv2.EVENT_LBUTTONDOWN:
		selectionPts[0] = (x, y)
		# gray_bkp = gray.copy()
		pState = "SELECTION_STARTED"

	# check if left mouse button was released during selection and mark selection finished
	elif event == cv2.EVENT_LBUTTONUP and pState == "SELECTING" or pState == "SELECTION_STARTED":
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		selectionPts[1] = (x, y)
		pState = "SELECTION_FINISHED"

# img: image patch to auto-correlate
# motion_window: (row, column) to compute auto-correlation surface
def calcScoreMatrixSurface(img, motion_window = (40, 40)): 
	f32_image = img.astype(np.float32)
	score_surf= np.zeros((motion_window[0],motion_window[1]), dtype=np.float32)
	for u in [i-motion_window[0]/2 for i in range(motion_window[0])]: #row
		for v in [i-motion_window[1]/2 for i in range(motion_window[1])]: #columns
			# Use affine transforms for translation
			M = np.float32([[1,0,v],[0,1,u]]) 
			dst = cv2.warpAffine(f32_image,M,(f32_image.shape[1],f32_image.shape[0]))
			norm_corr = -1 * sum(sum((f32_image.astype(np.float32) - np.mean(f32_image))*(dst.astype(np.float32) - np.mean(dst))/np.sqrt(np.var(f32_image.astype(np.float32))*np.var(dst))))
			score_surf[u+motion_window[0]/2][v+motion_window[1]/2] = norm_corr
	return score_surf

def plotErrSurface(img, mat):

	X = [i-mat.shape[1]/2 for i in range(mat.shape[1])]
	Y = [i-mat.shape[0]/2 for i in range(mat.shape[0])]

	X, Y = np.meshgrid(X, Y)
	fig = plt.figure()
	ax = fig.add_subplot(2,1,2,projection='3d') #fig.gca(projection='3d')
	
	surf = ax.plot_surface(X, Y, mat, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
	ax.set_zlim(np.min(mat), np.max(mat))
	fig.colorbar(surf, shrink=0.5, aspect=10)
	plt.title("Auto-correlation Error Surface")
	# ax.set_zscale('log')
	ax.set_ylabel("Row Motion Vector (u) Length")
	ax.set_xlabel("Column Motion Vector (v) Length")
	ax.set_zlabel("Normalized Cross-Correlation Error -ve")
	plt.subplot(2,1,1)
	plt.imshow(img, cmap='gray')
	plt.title("Image Patch")
	plt.show()


cv2.namedWindow("image")
cv2.setMouseCallback("image", onMouseEvent)
cap = cv2.VideoCapture(0)

while(True):

	# Capture frame-by-frame
	ret, frame = cap.read()

	# Stop capturing frames while selecting
	if pState != "SELECTING" and pState != "SELECTION_FINISHED":
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Create a back up of interest frame
	if pState == "SELECTION_STARTED":
		gray_bkp = gray.copy()

	# Draw rectangle 
	if pState == "SELECTING":
		gray = gray_bkp.copy()
		# draw a rectangle around the region of interest
		if selectionPts[1][0] + selectionPts[1][0] != -2:
			cv2.rectangle(gray, selectionPts[0], selectionPts[1], 255, 2)

	# Process patch
	elif pState == "SELECTION_FINISHED":
		if selectionPts[0][1] >= selectionPts[1][1] or selectionPts[0][0] >= selectionPts[1][0]: # keeping it simple
			pState = "IDLE"
			continue

		gray_bkp = gray[selectionPts[0][1]:selectionPts[1][1], selectionPts[0][0]:selectionPts[1][0]].copy() 
		cv2.rectangle(gray, selectionPts[0], selectionPts[1], 255, 5)
		
		# debug
		gray_bkp = np.zeros((100,100),dtype=np.uint8)
		gray_bkp[40:, :] = 255

		cv2.imshow('patch',gray_bkp)
		scoreMat = calcScoreMatrixSurface(gray_bkp, (gray_bkp.shape[0],gray_bkp.shape[1]))
		plotErrSurface(gray_bkp, scoreMat)
		pState = "IDLE"

	# Display the resulting frame
	cv2.imshow('image',gray)

	# exit on 'q'
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
	    break

	# reset selection on 'r'
	if key == ord("r"):
		selectionPts = [ (-1,-1) for i in range(2) ]
		pState = "IDLE"
		
# Release the capture
cap.release()
cv2.destroyAllWindows()

