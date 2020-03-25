import streamlit as st
import numpy as np
from PIL import Image

from utils import GetInputImage

from ISR.models import RDN, RRDN

#@st.cache
def make_model(name):
    l_model_name = ['gans', 'psnr-large', 'psnr-small', 'noise-cancel']
    if name not in l_model_name:
        return None
    else:
        model_fun = RRDN if l_model_name.index(name) == 0 else RDN
        return model_fun(weights = name)

#@st.cache
def sr_image(img_np_arr, model = RRDN(weights = 'gans')):
    '''
    Use ISR model to super-res the give image and return the resulting np array
    Args:
        model: currently 4 models are available
                RDN - psnr-large, psnr-small, noise-cancel
                RRDN - gans
    '''
    sr_im = model.predict(img_np_arr)
    return sr_im

def Main():
    app_doc = '''
    Super-Res your image using Residual Dense and Adversarial Network
    '''
    st.sidebar.header('ISR')
    st.sidebar.markdown(app_doc)

    #User Input
    color_channels = 'RGB'
    img_np_arr = GetInputImage(st_asset = st.sidebar, color = color_channels)
    l_model_name = ['gans', 'psnr-large', 'psnr-small', 'noise-cancel']
    model_name = st.sidebar.selectbox('model type', options = l_model_name)
    st.sidebar.markdown('[-> details](https://github.com/idealo/image-super-resolution#pre-trained-networks)')

    if type(img_np_arr) == np.ndarray:
        model = make_model(name = model_name)
        sr_img = sr_image(img_np_arr, model)

        print(f'org img shape: {img_np_arr.shape}\nSR img shape: {sr_img.shape}')

        h, w, c = img_np_arr.shape
        org_img = Image.fromarray(img_np_arr).resize(size = (h * 2, w * 2))
        new_img = Image.fromarray(sr_img).resize(size = (h * 2, w * 2))
        np_black_line = np.zeros(shape = (h,2,c))
        img_comb = np.hstack([
                            np.asarray(org_img)[:, :w, :],
                            np_black_line,
                            np.asarray(new_img)[:, w:, :]
                            ])
        st.subheader('Original vs Super Res')
        st.image(img_comb, channels = color_channels, use_column_width = True)

        # st.subheader('Original')
        # st.image(img_np_arr, channels = color_channels, use_column_width = True)
        #
        # st.subheader('Super Res')
        # st.image(sr_img, channels = color_channels, use_column_width = True)


if __name__ == '__main__':
	Main()
