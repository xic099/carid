import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
from scipy.optimize import minimize
from skimage.transform import resize
from matplotlib.image import imread
from PIL import Image

def sparse_reg(xd, eps=0.001):
    def L1_norm(x):
        return np.linalg.norm(x,ord=1)
    constr = ({'type': 'ineq', 'fun': lambda x:  eps - np.linalg.norm(Theta @ x - xd,2)})
    x0 = np.linalg.pinv(Theta) @ xd
    res = minimize(L1_norm, x0, method='SLSQP',constraints=constr)
    s1 = res.x
    return s1

def predict():
    scores = []
    current_index = 0
    for step in ncars:
        end_index = current_index + step
        end_index = min(end_index, len(s1))
        scores.append(sum(s1[current_index:end_index]))
        current_index += step
    predicted_label = car_type_counts[np.argmax(scores)][0]
    return predicted_label

st.set_page_config(layout='wide')  # Choose wide mode as the default setting

sidebar_style = """
    <style>
    .sidebar-result { color: green; font-size: 46px; }
    </style>
"""

# 添加样式

# Add a logo (optional) in the sidebar
logo = Image.open(r'logo.jpg')
st.sidebar.image(logo, width=180)

# Add the expander to provide some information about the app
with st.sidebar.expander("关于这个项目"):
    st.write("""
        该平台可以实现车辆自动识别，上传车辆照片，即可识别车辆类型
     """)
st.markdown(sidebar_style, unsafe_allow_html=True)
waiting_placeholder = st.sidebar.empty()
# Add an app title. Use css to style the title
st.markdown(""" <style> .font {                                          
    font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">车辆识别平台</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("选择图片文件", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    # 使用PIL读取上传的文件
    image = Image.open(uploaded_file)
    image_resized = image.resize((1200, 1200), Image.ANTIALIAS)

    waiting_message = "请等待，正在识别中..."
    waiting_placeholder.markdown(waiting_message)

    st.image(image_resized, caption='上传的图片',    use_column_width=True)

    # 车辆识别

    ncars =[16, 16, 17, 11, 11, 20, 17, 10, 19]
    car_type_counts = [('528', 16),
                         ('529', 16),
                         ('dahu', 17),
                         ('e3', 11),
                         ('e3l', 11),
                         ('g01', 20),
                         ('g05', 17),
                         ('x30', 10),
                         ('x30l', 19)]
    npy_file_path = 'theta_9_14.npy'
    Theta = np.load(npy_file_path)
    image = imread(uploaded_file)
    xd = resize(image, (9, 14), anti_aliasing=True)
    xd = np.mean(xd, -1)
    xd =xd.flatten()
    s1 = sparse_reg(xd)
    predicted_label = predict()

    result_message = "识别结果：**{}**".format(predicted_label)
    waiting_placeholder.markdown(result_message)

else:
    st.write("请上传图片文件。")