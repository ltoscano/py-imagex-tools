#!/usr/bin/python
# License: Creative Commons Zero http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
title           :OpenCVQuickFilters.py
description     :Quick utility to configure, apply and visualize OpenCV image filters in action
usage           :python OpenCVQuickFilters.py <image_path> [<location as x,y>, <size as heightxwidth>]
python_version  :2.7.6
depends         :numpy, opencv, matplotlib
author          :Yagnesh Revar
"""

try:
	import cv2
	import sys
	import numpy as np
	from matplotlib import pyplot as plt
	import math
	import os
	from scipy import stats
except ImportError:
	print "Something's wrong with the imports..:("

# Sample Open CV Abstractions 

# boxfilter(img, ksize, ddepth=-1, weights_array=None)
# usage:
# 	boxfilter(3, [1./x for x in range(1,10)])
# 	boxfilter(3)
# 	boxfilter(3, range(1,10)) 
def boxfilter(img, ksize, ddepth=-1, weights_array=None):
	if weights_array is not None:
		if sum(weights_array) == 1.0:
			weights = np.matrix(weights_array, np.float32).reshape(ksize,ksize)
		else:
			weights = (np.matrix(weights_array, np.float32)/float(sum(weights_array))).reshape(ksize,ksize)
	else:
		weights = np.ones((ksize,ksize),np.float32)/(ksize**2)

	return cv2.filter2D(img,ddepth,weights)

def median(img, ksize):
	return cv2.medianBlur(img, ksize)

def sobel_wrap(*args, **kwargs):
	if args:
		return cv2.Sobel(*args)
	elif kwargs:
		return cv2.Sobel(**kwargs)

def sobel(img, ksize=3, ddepth=cv2.CV_64F, xorder=1, yorder=1):
	return sobel_wrap(img, ddepth, xorder, yorder, apertureSize=ksize)

def scharr_x(img, ddepth=cv2.CV_64F):
	return sobel_wrap(img, ddepth, 1, 0, -1)

def scharr_y(img, ddepth=cv2.CV_64F):
	return sobel_wrap(img, ddepth, 0, 1, -1)

def laplacian(img, ddepth=cv2.CV_64F, ksize=3):
	return cv2.Laplacian(img,ddepth, ksize=ksize)

def absolute(img):
	return np.abs(img)

def touint8(img):
	return img.astype('uint8')

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
erode_kernel = np.ones((3,3),np.uint8)
erode_iter = 1

filters = [
			# filters 

				[['Mean 3x3 8u', boxfilter, 3, cv2.CV_8U]],
				[['Mean 5x5 8u', boxfilter, 5, cv2.CV_8U]],
				# ['Mean 10x10 8u', boxfilter,10, cv2.CV_8U],
				# ['Median 5x5 8u', median,5],
				[['Median 25x25 8u', median, 25]],
				# ['Scharr x 3x3 16s', scharr_x, cv2.CV_16S],
				# ['Scharr y 3x3 16s', scharr_y, cv2.CV_16S],
				[['Dilate 3x3 iter 1', cv2.erode, erode_kernel, erode_iter]],
				[['Erode 3x3 iter 1', cv2.dilate, erode_kernel, erode_iter]],

			# filter chains
				['Sobel xy o-1 3x3 16s ', 
					['', sobel, 3, cv2.CV_16S, 1, 1],
					['', absolute],
					['', touint8]
					],

				['Sobel xy o-2 3x3 16s', 
					['', sobel, 3, cv2.CV_16S, 2, 2],
					['', absolute],
					['', touint8]
					],
			
				[
					['Mean 3x3 8u', boxfilter, 3, cv2.CV_8U],
					['Laplacian 3x3 16s', laplacian, cv2.CV_16S, 5],
					['', absolute],
					# ['', touint8]
					
					],	
		  ]

plot_cols = int(math.floor(math.sqrt(len(filters)))) + 1 
plot_rows = int(math.ceil(math.sqrt(len(filters)))) 
curr_plot = 0

def addSubplot(displayImage, title=''):
	global curr_plot
	curr_plot += 1; plt.subplot(plot_cols, plot_rows, curr_plot)
	plt.imshow(displayImage, cmap='gray'); 
	plt.title(title); 
	plt.axis('off')

def applyFilters(img, filters, displayFn, f_id = 0, chain = -1): 

	if f_id < len(filters):

		if chain == -1:
			filtered_img = img
			if type(filters[f_id][0]) == str:
				fchain_name = filters[f_id][0]
			else:
				fchain_name = filters[f_id][0][0]

			print "Exploring chain: ", fchain_name

			# process chains recursively
			for chain_no in range(len(filters[f_id])):
				if type(filters[f_id][chain_no]) == list: # validate a filter 
					filtered_img = applyFilters(filtered_img, filters, displayFn, f_id, chain_no)

			# display results
			displayFn(filtered_img, fchain_name)
			applyFilters(img, filters, displayFn, f_id+1)
		else:
			# process filter
			print "   Applying filter -> ", filters[f_id][chain][0], "..."
			filtered_img = filters[f_id][chain][1](img, *filters[f_id][chain][2:])
			return filtered_img

def main(args):

	global curr_plot

	# parse arguments
	try:
		idx = 1
		img_file = args[idx]
	except:
		img_file = None

	if img_file:

		# parse patch location arg
		idx+=1 
		try:
			img_loc = map(int, args[idx].split(','))
		except:
			img_loc = None

		# parse patch size arg
		idx+=1 
		try:
			img_size = map(int, args[idx].split('x'))
		except:
			img_size = None

		print("Got image=%s, location=%s, size=%s"%(img_file, img_loc, img_size))


		img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
		
		if img is None:
			print "Invalid image path supplied"
			return
		

		if img_loc and img_size:
			img = img[img_loc[0]:img_loc[0]+img_size[0], img_loc[1]:img_loc[1]+img_size[1]]
		elif img_loc:
			img = img[img_loc[0]:, img_loc[1]:]
	else:
		test_img = np.uint8(np.ones((90,90))*-1)
		test_img[30:60,30:60] = 0
		img = test_img


	print "Image shape ", img.shape

	# Conf plots
	print plot_cols, plot_rows

	# plt.rcParams['figure.figsize'] = (10.0, 10.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['figure.autolayout'] = True
	# plt.rcParams['image.cmap'] = 'gray'

	plt.figure()

	curr_plot += 1; 
	plt.subplot(plot_cols, plot_rows, curr_plot)
	plt.imshow(img, cmap='gray'); plt.title("Original Image")
	plt.axis('off')

	# Iterate over filter graph and display
	applyFilters(img, filters, addSubplot)

	plt.tight_layout()		
	plt.show()

if __name__ == '__main__':
    main(sys.argv)