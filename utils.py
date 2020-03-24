import streamlit as st
import numpy as np
import imageio

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
