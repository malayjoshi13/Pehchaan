# Pehchaan
Face detection, verification and recognition/identification system

## 1) Install

- Create a virtual environment named "pehchaan" (only once):

  `conda create -n pehchaan`

- Activate the virtual environment each time:

  `conda activate pehchaan`

- Install dependencies (only once):

  `conda install pip`

  `pip install -r requirements.txt`
  
## 2) Folder structure

After cloning this GitHub repository, setup the database inside `dataset` folder by taking reference of following folder structure
```
cloned version of Pehchaan github repo
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
You can anytime add/delete images in the database, the representation files will be made automatically for `FaceRecognizer` task

## 3) Execution

## 3.1) FaceRecognizer:-
 for identifying name of person (whose image is feeded as an input) out of the database (i.e. `dataset` folder)

Syntax:- `python FaceRecognizer.py target_image_path database_path face_detector_model face_recognizer_model distance_metric`

**Behind the hood**: applies SimilarFaceFinder algorithm in 1:N format input image and each image of database. Whichever database image outputs to be similar to input image, its label is considered to be identity of the input image. In case where multiple database images are found to be similar to input image, in such case label with maximum occurence is considered as label of the input image.

Ex:- `python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset mtcnn VGG-Face euclidean` OR  `python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset` as by default, "retinaface" is used as detector and "ArcFace" as recognizer and "euclidean" as distance metric

## 3.2) SimilarFaceFinder:-
 for finding all faces (present in the database) which look similar to target face input to the system

**Behind the hood**: applies FaceVerifier algorithm in 1:N format between input image and each database image to check distance between feature vectors of which database image and input image is less than the threshold value. Featured vector of whichever database image satisfy this condition is considered similar to the input image.

Syntax:- `python SimilarFaceFinder.py target_image_path database_path face_detector_model face_recognizer_model distance_metric`

Ex:- `python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset mtcnn VGG-Face euclidean` OR `python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset` as by default, "retinaface" is used as detector and "ArcFace" as recognizer and "euclidean" as distance_metric

## 3.3) FaceVerifier:-
 for verifying if two input images different or same.

**Behind the hood**: do 1:1 checking to find how close feature vectors of two input images are.

Syntax:- `python SimilarFaceFinder.py target_image1_path target_image2_path face_detector_model face_recognizer_model distance_metric`

Ex:- `python FaceVerifier.py ./dataset/kalam/1.jpg ./dataset/kalam/hi.jpeg mtcnn VGG-Face euclidean` OR `python FaceVerifier.py ./dataset/kalam/1.jpg ./dataset/kalam/hi.jpeg` as by default, "retinaface" is used as detector and "ArcFace" as recognizer and "euclidean" as distance_metric

## 3.4) FaceDetector:-
 finds face region out of the whole input image of a person/identity

Syntax:- `python FaceDetector.py ./dataset/kalam/1.jpg face_detector_model`

Ex:- `python FaceDetector.py target_image_path mtcnn` OR `python FaceDetector.py ./dataset/kalam/1.jpg` as by default it has "retinaface"

## 4) Options to choose from

1) For `face_detector_model`:- opencvhaar, opencvdnn, dlib, mtcnn, retinaface, mediapipe
2) For `face_recognizer_model`:- VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, SFace

## 5) Use cases

1) auto-tagging individual(s) out of image(s),
2) tracking shoppers in cashierless shops/stores
3) tracking movement of a specific criminal/suspect/individual
4) smart attendance/check-in/entry in event/office/college/home/school
5) biometric/phone/system authentication, etc
