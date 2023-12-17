import streamlit as st 
from pathlib import Path
import os
import time

if 'Name' not in st.session_state:
    st.session_state['Name'] = "" 
if 'Submit' not in st.session_state:
    st.session_state['Submit'] = None
if 'reference_images_not_saved' not in st.session_state:
    st.session_state['reference_images_not_saved'] = True


len_imgs_in_save_folder = 0
ReferenceImage_dict = dict()


# asking user for label/identity/name. 
Name = st.text_input("Enter name of the person and press enter: ", key = "label")
if not st.session_state['Name']:
    st.session_state['Name'] = Name 


# as name is entered by user then ask for 4 reference images.
if st.session_state['Name'] and Name!="": 
    save_folder = './dataset/'+str(Name)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    len_images_in_save_folder = len(os.listdir(save_folder))

    # count number of images already in label's folder.
    for file in os.listdir(save_folder):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            len_imgs_in_save_folder+=1

    # if 4 reference images already exist, then show this message.
    if len_imgs_in_save_folder == 4:
        st.write("Already 4 reference images exist for this person. No need to upload more reference image(s).")
        
    # if 4 reference images don't exist, then ask user for left out reference image per label from user. 
    while len_imgs_in_save_folder<4:
        len_imgs_in_save_folder += 1
        ReferenceImage = st.file_uploader(label = "Upload reference image of this person", type=["png","jpg", "jpeg"], key =f"image_{len_imgs_in_save_folder}")
        if ReferenceImage:
            ReferenceImage_dict[ReferenceImage.name]= ReferenceImage.getvalue()

        # when 4th reference image is given by user then show submit button
        if len_imgs_in_save_folder==4 and ReferenceImage:
            submit = st.button('Save these reference image(s)!')
            if not st.session_state['Submit']:
                st.session_state['Submit'] = submit     

            # if submit is clicked, then save reference image to "dataset" folder in sub-folder corresponding to label/name of person in it and show success message and rerun.
            if st.session_state['Submit']:
                save_folder = './dataset/'+str(Name)
                Path(save_folder).mkdir(parents=True, exist_ok=True)
                
                for ReferenceImage_name in ReferenceImage_dict.keys():
                    save_path = str(save_folder) + "/" + str(ReferenceImage_name)
                    with open(save_path, mode='wb') as w:
                        w.write(ReferenceImage_dict[ReferenceImage_name])

                st.success(f'Reference images for {st.session_state["Name"]} are successfully saved at "{save_folder}" folder!')  
                time.sleep(8)
                st.rerun()