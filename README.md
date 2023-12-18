# Pehchaan
Face detection and recognition using pre-trained Deep and Shallow architecture models.

Face detection → Finding face out of a full image
Face recognition → Tell name of face in a given image

## 1) Setting up work environment
```
git clone https://github.com/malayjoshi13/Pehchaan.git

conda create -n pehchaan

conda activate pehchaan

pip install -r requirements.txt
```
  
## 2) Directory structure

After cloning this GitHub repository, set the database inside the `dataset` folder by taking reference of the following folder directory structure:
```
Pehchaan
├── dataset folder
│   ├── Name of Person1
│   │   ├── First image of Person1
│   │   ├── First image of Person1
│   ├── Name of Person2
│   │   ├── First image of Person2
.   .   .
.   .   .
.   .   .
.   .   .
```
Note: You can add/delete images in the database anytime, the representation files will be made automatically for the `FaceRecognizer` functionality

## 3) Features

## 3.1) FaceRecognizer:-
**Application**: identifying the name of a person whose image is fed as an input using images in the database (i.e. `dataset` folder). 

**Syntax**: `python FaceRecognizer.py target_image_path database_path face_detector_model face_recognizer_model distance_metric`

**Under the hood**: `FaceRecognizer` algorithm applies the `SimilarFaceFinder` algorithm in 1:N format on the input image and each image of the database to find all database images which are similar to the input image. After this, the `FaceRecognizer` algorithm selects the most common/repetitive/famous label out of labels of these matched/found database images. This selected label is considered to be the identity of the input image. 

**Ex 1**: `python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset MTCNN VGGFace euclidean` --> here "MTCNN" is used as face detector and "VGGFace" as face recognizer and "euclidean" as distance_metric

**Ex 2**: `python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset` --> here by default, "RetinaFace" is used as face detector, "ArcFace" as face recognizer and "euclidean" as the distance metric

```Note: Rather than fine-tuning the pre-trained face recognition models, I choose to calculate the distance between feature vectors of input image and images in database. This is because having a small database in this use-case, fine-tuning didn't give the expected results (shown by results between two approaches below).```

## 3.2) SimilarFaceFinder:-
**Application**: finding all database images which look similar to the face in the input image.

**Under the hood**: `SimilarFaceFinder` algorithm applies the `FaceVerifier` algorithm in 1:N format between the input image and each database image to find the database image and input image having a distance (between their corresponding feature vectors) less than the threshold value. Whichever database image satisfies this condition is considered similar to the input image.

Syntax:- `python SimilarFaceFinder.py target_image_path database_path face_detector_model face_recognizer_model distance_metric`

**Ex 1**: `python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset MTCNN VGGFace euclidean` --> here "MTCNN" is used as face detector and "VGGFace" as face recognizer and "euclidean" as distance_metric

**Ex 2**: `python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset` --> here by default, "RetinaFace" is used as face detector, "ArcFace" as face recognizer and "euclidean" as distance metric

## 4) Models used

4.1) For face detection:- <br>
a) OpenCVHaar <br>

b) OpenCVDNN <br>
 
c) Dlib <br>

d) MTCNN <br>

e) RetinaFace --> https://insightface.ai/retinaface | official project page <br>

f) MediaPipe <br>

Weights for the face detector model get automatically downloaded from the internet; nothing is required from our end.

4.2) For face recognition:- <br>
a) VGGFace --> https://www.robots.ox.ac.uk/~vgg/software/vgg_face/ | Official project page. <br>

b) OpenFace --> https://cmusatyalab.github.io/openface/ | It's an unofficial Pytorch implementation of https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf developed at Google. <br>

c) Facenet --> https://github.com/davidsandberg/facenet | It's an unofficial Tensorflow implementation of https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf developed at Google and took reference from OpenFace library. <br>

d) DeepFace --> https://github.com/swghosh/DeepFace | It's an unofficial Tensorflow implementation of https://research.facebook.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/ developed at Meta. <br>
`Note`: Don't be confused between DeepFace model developed at Meta (mentioned above) and [deepface](https://github.com/serengil/deepface) library (with wrappers of many face recognition and detection models) developed by Serengil. <br>

e) DeepID --> https://github.com/kamwoh/face_recognition | It's an unofficial implementation of https://dl.acm.org/doi/10.1109/CVPR.2014.244 (DeepID 1 model). It has more variants like DeepID 2, DeepID 2+, DeepID 3. <br> 

f) ArcFace --> https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch | Official implementation of https://arxiv.org/abs/1801.07698. The same authors have also developed RetinaFace model, [insightface](https://github.com/deepinsight/insightface) library for face detection and recognition and a few other models (see here: https://insightface.ai/projects). <br>

g) SFace --> https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface (unofficial) and https://github.com/zhongyy/SFace (official) implementations of https://arxiv.org/abs/2205.12010.<br>

`Please refer to https://arxiv.org/abs/1804.06655 to see the accuracy of different face recognition methods, including those we have used in this project.`

## 5) Working on

5.1) Understanding each face detection and recognition model used in this project in `algos.md` file and adding what learned about each of them in Gdrive's folder _Studying > CV. <br>

5.2) Conducting comparative analysis between models by running them in different pairs to see which detector-recognition model pair works the best.

5.3) Analysing how fine-tuning face recognition models creates better representations of faces passed to it.

5.4) Extend this project to another use case like from given multiple CCTV videos, locating different positions where a person (missing/thief) was present at a particular time using his/her single image. Thus saves time to manually view the whole CCTV footage. Another possible use case is tracking movement of all customers just by their faces within a shopping place to enable the experience of cashier-free shopping experience.

5.5) Adding illumination robustness, varying poses robustness and face-occulsion robustness on top of existing face detection and recognition-based pipeline.
6.7) restructuring model structure to include required Fully Connected layers which have demonstrated better understanding of minute differences between face features. Also including required cost functions that can help in discriminative learning by increasing difference between non-similar classes.

## 6) References

6.1) PARKHI et al. "DEEP FACE RECOGNITION" --> https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf <br>

6.2) Masi, Iacopo et al. “Deep Face Recognition: A Survey" --> https://www.semanticscholar.org/paper/Deep-Face-Recognition%3A-A-Survey-Masi-Wu/8de1c724a42d204c0050fe4c4b4e81a675d7f57c  <br>

6.3) Wang, Xinyi et al. “A Survey of Face Recognition” --> https://www.semanticscholar.org/paper/A-Survey-of-Face-Recognition-Wang-Peng/80502a95ab1f83e07febf82c552ae12fcecab00a  <br>
