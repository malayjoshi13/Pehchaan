# Pehchaan

Pehchaan is a one-shot labeling tool to identify the name of the person present in an image. <br>

A potential use case around which I have started to work on this project is to automate the process of labeling the people present in pictures and photographs having immense significance. In the absence of such an automated process, it's a very time-consuming and labor-intensive task to manually label people present in a large stock of photographs at digital libraries across India as well as the globe. Without these labels, these significant documents are mere pieces of memory/space-consuming items, nothing more. <br>

Broadly, this tool makes use of pre-trained Face detection (for finding face out of a full image), Face alignment, and Face recognition (for generating discriminating feature vectors for each passed face image) models and algorithms to keep checking if the database (collection of reference images to perform feature matching algorithm) is modified and doing one-to-one matching between feature representation of image input by user and image(s) in database. <br>

This work is representative of work done as part of my internship at DESIDOC-DRDO (New Delhi, India) and has no direct association with the full work done during the internship period. 

## Getting started
```
git clone https://github.com/malayjoshi13/Pehchaan.git

conda create -n pehchaan

conda activate pehchaan

pip install -r requirements.txt

streamlit run Home.py
```

## Features

1) **Create a database of reference images of people** you strongly believe could be in the pile of photographs you'll be auto-labeling. Not able to think of all possible people in starting itself? Not an issue, keep adding reference images as you keep remembering. At the back, the pipeline will keep creating required representations every time you add any new image of a particular person to be used for face matching during the auto-labeling process later.

2) **Auto-label pile of photographs/pictures** you have by just passing one photograph at a time and getting the names of all people in that photograph.

## Architecture


## Models used

1) `For face detection`:- <br>

a) OpenCVHaar --> model wrapper that makes use of OpenCV's CascadeClassifier function and pre-trained Haar cascade models (see here: https://github.com/opencv/opencv/tree/master/data/haarcascades). <br>

b) OpenCVDNN --> model wrapper that makes use of Caffe-based face detector present in the face_detector sub-directory of the [OpenCV DNN module](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector). <br>
 
c) MTCNN --> model wrapper that makes use of unofficial [tensorflow implementation](https://github.com/ipazc/mtcnn) of https://ieeexplore.ieee.org/document/7553523 work done by K. Zhang, Z. Zhang, Z. Li and Y. Qiao. <br>

d) RetinaFace --> model wrapper that makes use of https://insightface.ai/retinaface | official project page. <br>

e) MediaPipe --> model wrapper that makes use of the [face detection tool](https://developers.google.com/mediapipe/solutions/vision/face_detector)which is part of Mediapipe developed at Google. <br>

Weights for the face detector model get automatically downloaded from the internet; nothing is required from our end.
<br><br><br>

2) `For face recognition`:- <br>

a) VGGFace --> https://www.robots.ox.ac.uk/~vgg/software/vgg_face/ | Official project page. <br>

b) OpenFace --> https://cmusatyalab.github.io/openface/ | It's an unofficial Pytorch implementation of https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf developed at Google. <br>

c) Facenet --> https://github.com/davidsandberg/facenet | It's an unofficial Tensorflow implementation of https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf developed at Google and took reference from OpenFace library. <br>

d) DeepFace --> https://github.com/swghosh/DeepFace | It's an unofficial Tensorflow implementation of https://research.facebook.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/ developed at Meta. <br>
`Note`: Don't be confused between DeepFace model developed at Meta (mentioned above) and [deepface](https://github.com/serengil/deepface) library (with wrappers of many face recognition and detection models) developed by Serengil. <br>

e) DeepID --> https://github.com/kamwoh/face_recognition | It's an unofficial implementation of https://dl.acm.org/doi/10.1109/CVPR.2014.244 (DeepID 1 model). It has more variants like DeepID 2, DeepID 2+, DeepID 3. <br> 

f) ArcFace --> https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch | Official implementation of https://arxiv.org/abs/1801.07698. The same authors have also developed RetinaFace model, [insightface](https://github.com/deepinsight/insightface) library for face detection and recognition and a few other models (see here: https://insightface.ai/projects). <br>

g) SFace --> https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface (unofficial) and https://github.com/zhongyy/SFace (official) implementations of https://arxiv.org/abs/2205.12010.<br>

`Please refer to https://arxiv.org/abs/1804.06655 to see the accuracy of different face recognition methods, including those we have used in this project.`

## Working on

1) Understanding each face detection and recognition model used in this project and adding what learned about each of them in Gdrive's folder _Studying > CV. <br>

2) Conducting comparative analysis between models on parameters such as accuracy, precision, and speed by running them in different pairs to see which detector-recognition model pair works the best. ([useful resource](https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c))

3) Analysing how fine-tuning face recognition models creates better representations of faces passed to it. This could potentially take the existing closed-data auto-labeling capability to open-data auto-labeling, where the chances of this tool labeling an image whose reference image is not there in the database, to be "Unknown" will be higher than existing scenario.

4) Extend this project to another use case like from given multiple CCTV videos, locating different positions where a person (missing/thief) was present at a particular time using his/her single image. Thus saves time to manually view the whole CCTV footage. Another possible use case is tracking movement of all customers just by their faces within a shopping place to enable the experience of cashier-free shopping experience.

5) Adding illumination robustness, varying poses robustness and face-occulsion robustness on top of existing face detection and recognition-based pipeline.

## References

6.1) PARKHI et al. "DEEP FACE RECOGNITION" --> https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf <br>

6.2) Masi, Iacopo et al. “Deep Face Recognition: A Survey" --> https://www.semanticscholar.org/paper/Deep-Face-Recognition%3A-A-Survey-Masi-Wu/8de1c724a42d204c0050fe4c4b4e81a675d7f57c  <br>

6.3) Wang, Xinyi et al. “A Survey of Face Recognition” --> https://www.semanticscholar.org/paper/A-Survey-of-Face-Recognition-Wang-Peng/80502a95ab1f83e07febf82c552ae12fcecab00a  <br>

## End-note
Thank you for patiently reading till here. I am pretty sure just like me, you would have also learned something new about developing different types of face detection and recognition models and a practical use case of an auto-labeling tool using these models. Using these learned concepts, I will push myself to continue improving this tool. I encourage you also to pick the tasks I have stated above to improve this tool!!

## Contributing
You are welcome to contribute to the repository with your PRs. In case of query or feedback, please write to me at 13.malayjoshi@gmail.com or https://www.linkedin.com/in/malayjoshi13/.
