import streamlit as st 
from utils import supported_detection_model, supported_recognition_model
from services import FaceRecognizer
from pathlib import Path
import os

if 'face_detection_model' not in st.session_state:
    st.session_state['face_detection_model'] = " "
if 'face_recognition_model' not in st.session_state:
    st.session_state['face_recognition_model'] = " "    
if 'do_auto_label' not in st.session_state:
    st.session_state['do_auto_label'] = None   
if 'user_image_saved' not in st.session_state:
    st.session_state['user_image_saved'] = False 

ReferenceImages = []
ReferenceImage_dict = dict()
len_imgs_in_save_folder = 0

st.title("Photograph Auto-labeler")


# if there is some reference image saved in "database" folder, then ask user to select face detection and recognition models
len_database = len(os.listdir('./database'))
if len_database > 0:
    
    face_detection_model = st.selectbox("Select face detection model:", supported_detection_model.final_list, key = "face detection model")
    face_recognition_model = st.selectbox("Select face recognition model:", supported_recognition_model.final_list, key = "face recognition model")
    st.session_state['face_detection_model'] = face_detection_model
    st.session_state['face_recognition_model'] = face_recognition_model

    # once user selects both face detection and recognition models, ask user to upload image to auto-label it
    if st.session_state['face_recognition_model'] != " " and st.session_state['face_detection_model'] != " ":
        UserImage = st.file_uploader(label = "Upload picture to be auto-labelled", type=["png","jpg", "jpeg"], key="user image uploader")
        if UserImage:
            st.session_state['user_image_saved'] = True
        
        # once user has also uploaded user image, show user "Auto-label" button
        if st.session_state['user_image_saved']:
            do_auto_label = st.button('Auto-label this image!')
            st.session_state['do_auto_label'] = do_auto_label
            

#.......................................................................................


# once user clicks on "Auto-label" button, user image will be saved and will auto-labeled.
if st.session_state['do_auto_label']:
    # user image will be saved
    save_user_folder = './temp/'
    save_user_path = str(save_user_folder) + "/" + str(UserImage.name)
    Path(save_user_folder).mkdir(parents=True, exist_ok=True)

    with open(save_user_path, mode='wb') as w:
        w.write(UserImage.getvalue())
        # st.success(f'User image {UserImage.name} is successfully saved at {save_user_folder}!')

    # user image will be auto-labeled
    result = FaceRecognizer.recognize(img_path = save_user_path, db_path = './database', face_recog_model = face_recognition_model, distance_metric = 'euclidean', face_detection_model = face_detection_model)
    st.write("Name of person in input image is:-", result)