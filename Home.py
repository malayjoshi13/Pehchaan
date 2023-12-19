# streamlit run Home.py

import streamlit as st 

st.title("Welcome to Pehchaan")

st.write('''Pehchaan is a one-shot labeling tool to auto-label photographs by identifying name of the person(s) present in them.
         
In absence of such an automated process, it's a very time-consuming and labor-intensive task to manually label people present in a large pile of photographs at digital libraries across India as well as the globe. Without these labels, these significant photographs are mere pieces of memory/space-consuming items, nothing more.

Broadly, this tool makes use of pre-trained Face detection (for finding face out of a full image), Face alignment, and Face recognition (for generating discriminating feature vectors for each passed face image) models and algorithms to keep checking if the database (collection of reference images to perform feature matching algorithm) is modified and doing one-to-one matching between feature representation of image input by user and image(s) in database.

Features of this tool are:
1. Create a database of reference images of people you strongly believe could be in the pile of photographs you'll be auto-labeling. Not able to think of all possible people in starting itself? Not an issue, keep adding reference images as you keep remembering. At the back, the pipeline will keep creating required representations every time you add any new image of a particular person to be used for face matching during the auto-labeling process later.

2. Auto-label pile of photographs/pictures you have by just passing one photograph at a time and getting the names of all people in that photograph.
         
         ''')
         
         
         
         
    