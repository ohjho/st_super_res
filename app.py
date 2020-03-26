import streamlit as st
import numpy as np
from PIL import Image

from utils import GetInputImage, image_hstack

from ISR.models import RDN, RRDN

#@st.cache
def make_model(name):
    l_model_name = ['gans', 'psnr-large', 'psnr-small', 'noise-cancel']
    if name not in l_model_name:
        return None
    else:
        model_fun = RRDN if l_model_name.index(name) == 0 else RDN
        return model_fun(weights = name)

@st.cache
def sr_image(img_np_arr, model_name = 'gans'):
    '''
    Use ISR model to super-res the give image and return the resulting np array
    Args:
        model: currently 4 models are available
                RDN - psnr-large, psnr-small, noise-cancel
                RRDN - gans
    '''
    model = make_model(model_name)
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
    left_pct = st.sidebar.slider('Percentage of Orginal Image to show:', min_value = 0.0, max_value = 1.0, step = 0.1, value = 0.5)

    if type(img_np_arr) == np.ndarray:
        sr_img = sr_image(img_np_arr, model_name = model_name)

        print(f'org img shape: {img_np_arr.shape}\nSR img shape: {sr_img.shape}')

        # h, w, c = sr_img.shape
        # org_img = Image.fromarray(img_np_arr).resize(size = (w, h))
        # new_img = Image.fromarray(sr_img)
        # black_line_width = 5
        # np_black_line = np.zeros(shape = [h,black_line_width, c], dtype = np.uint8)
        # img_comb = np.hstack([
        #                     np.asarray(org_img)[:, :int(w/2), :],
        #                     np_black_line,
        #                     np.asarray(new_img)[:, int(w/2):, :]
        #                     ])
        st.subheader('Original vs Super Res')
        img_comb = image_hstack(img_np_arr, sr_img, left_pct = left_pct, black_line_width = 5, up_scale = True)
        st.image(img_comb, channels = color_channels, use_column_width = True)

        # st.subheader('Original')
        # st.image(img_np_arr, channels = color_channels, use_column_width = True)
        #
        # st.subheader('Super Res')
        # st.image(sr_img, channels = color_channels, use_column_width = True)


if __name__ == '__main__':
	Main()
