# streamlit run app.py

import streamlit as st 
from utils import supported_detection_model, supported_recognition_model
from services import FaceRecognizer
from pathlib import Path
import os

if 'Submit' not in st.session_state:
    st.session_state['Submit'] = None
if 'reference_image_saved' not in st.session_state:
    st.session_state['reference_image_saved'] = False
if 'face_detection_model' not in st.session_state:
    st.session_state['face_detection_model'] = " "
if 'face_recognition_model' not in st.session_state:
    st.session_state['face_recognition_model'] = " "    
if 'do_auto_label' not in st.session_state:
    st.session_state['do_auto_label'] = None   
if 'user_image_saved' not in st.session_state:
    st.session_state['user_image_saved'] = False 

ReferenceImages = []
Name = None   
count = 0

st.title("Photograph Auto-labeler")


# asking user for label/identity/name 
Name = st.text_input("Name of the person: ", key = "label")
if Name:
    save_folder = './dataset/'+str(Name)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    len_images_in_save_folder = len(os.listdir(save_folder))
    for file in os.listdir(save_folder):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            count+=1
    len_save_folder = count
    print(len_save_folder)

    # asking user for 4 reference image per label. If the label folder already has 4 reference images, then dont allow more uploads.
    while len_save_folder<4:
        len_save_folder += 1
        ReferenceImages = st.file_uploader(label = "Upload reference image of this person", type=["png","jpg", "jpeg"], key =f"image_{len_save_folder}", accept_multiple_files=True)
    # if name and reference image are given by user, then show submit button
    if Name and ReferenceImages:
        Submit = st.button('Submit')
        if not st.session_state['Submit']:
            st.session_state['Submit'] = Submit      


# if submit is clicked, then save reference image to dataset folder in sub-folder corresponding to label/name of person in it & show option to select models
if st.session_state['Submit']:
    save_folder = './dataset/'+str(Name)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    for ReferenceImage in ReferenceImages:
        save_path = str(save_folder) + "/" + str(ReferenceImage.name)
        with open(save_path, mode='wb') as w:
            w.write(ReferenceImage.getvalue())
            # st.success(f'Reference image {ReferenceImage.name} is successfully saved at "dataset" folder!')
            if not st.session_state['reference_image_saved']:
                st.session_state['reference_image_saved'] = True


# once reference image is saved, show options of models to select   
if st.session_state['reference_image_saved']:
    face_detection_model = st.selectbox("Select face detection model:", supported_detection_model.final_list, key = "face detection model")
    face_recognition_model = st.selectbox("Select face recognition model:", supported_recognition_model.final_list, key = "face recognition model")
    st.session_state['face_detection_model'] = face_detection_model
    st.session_state['face_recognition_model'] = face_recognition_model

    # once user selects both face detection and recognition models, ask user to upload image to auto-label it
    if st.session_state['face_recognition_model'] != " " and st.session_state['face_detection_model'] != " ":
        UserImage = st.file_uploader(label = "Upload file", type=["png","jpg", "jpeg"], key="user image uploader")
        if UserImage:
            st.session_state['user_image_saved'] = True
        
        # once user has also uploaded user image, show user "Auto-label" button
        if st.session_state['user_image_saved']:
            do_auto_label = st.button('Auto-label this image!')
            st.session_state['do_auto_label'] = do_auto_label


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
    result = FaceRecognizer.recognize(img_path = save_user_path, db_path = './dataset', face_recog_model = face_recognition_model, distance_metric = 'euclidean', face_detection_model = face_detection_model)
    st.write("Identity of person in input image is:-", result)