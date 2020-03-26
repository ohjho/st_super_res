import streamlit as st
import numpy as np
import imageio
from PIL import Image

def GetInputImage(st_asset = st, type = ['jpg', 'png', 'jpeg'], color = 'RGB'):
	'''
	Ask for user's input in the st_asset and return a NP array using imageio
	'''
	image_url = st_asset.text_input("Enter Image URL")
	image_fh = st_asset.file_uploader(label = "Or Upload your file", type = ['jpg', 'jpeg', 'png'])

	if image_url or image_fh:
		img = imageio.imread(image_url) if image_url else imageio.imread(image_fh)
		img = np.asarray(img)

		if color == 'BGR':
			img = img[:,:,::-1]
		elif color == 'RGB':
			pass
		else:
			img = None
		return img
	else:
		return None

def image_hstack(left_img_arr, right_img_arr, left_pct = None, black_line_width = 5, up_scale = False, resample = None):
	'''
	return one image of left_img_arr and right_img_arr join together
	Args:
		left_img_arr: must have same aspect ratio as right_img_arr
		left_pct: Percentage of left image to show, right image will fill up the remainder. If none, both left and right images will be shown in full
		up_scale: if left and right images are different in size, use the bigger w & h
	'''

	h, w, __ = left_img_arr.shape
	_h, _w, __ = right_img_arr.shape
	if up_scale:
		h, w = max(h, _h), max(w, _w)
	else:
		h, w = min(h, _h), min(w, _w)

	if left_pct > 1:
		raise TypeError(f"left_pct must be float and less than or equal to 1.")
	left_end = left_pct if left_pct else 1
	right_start = left_pct if left_pct else 0

	l_img = Image.fromarray(left_img_arr).resize(size = (w,h), resample = resample)
	r_img = Image.fromarray(right_img_arr).resize(size = (w,h), resample = resample)
	np_black_line = np.zeros(shape = [h, black_line_width, 3], dtype = np.uint8)
	img_comb = np.hstack([
						np.asarray(l_img)[:, :int(w * left_end ), :],
						np_black_line,
						np.asarray(r_img)[:,int(w * right_start):,:]
						])
	return img_comb
