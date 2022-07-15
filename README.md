# Pehchaan
Face recognizer system

## Install

- Create a virtual environment named "drdo" (only once):

  `conda create -n drdo`

- Activate the virtual environment each time:

  `conda activate drdo`

- Install dependencies (only once):

  `conda install pip`

  `pip install -r requirements.txt`
  
## Folder structure

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

## Execution

1) FaceRecognizer:- for identifying name of person (whose image is feeded as an input) out of the database (i.e. `dataset` folder)

Syntax:- `python FaceRecognizer.py target_image_path database_path face_detector_model face_recognizer_model distance_metric`

Ex:- `python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset mtcnn VGG-Face euclidean` OR  `python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset`         as by default, "retinaface" is used as detector and "ArcFace" as recognizer and "euclidean" as distance metric

2) SimilarFaceFinder:- for finding all faces (present in the database) which look similar to target face input to the system

Syntax:- `python SimilarFaceFinder.py target_image_path database_path face_detector_model face_recognizer_model distance_metric`

Ex:- `python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset mtcnn VGG-Face euclidean` OR `python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset` as by default, "retinaface" is used as detector and "ArcFace" as recognizer and "euclidean" as distance_metric

3) FaceVerifier:- for verifying if two input images different or same

Syntax:- `python SimilarFaceFinder.py target_image1_path target_image2_path face_detector_model face_recognizer_model distance_metric`

Ex:- `python FaceVerifier.py ./dataset/kalam/1.jpg ./dataset/kalam/hi.jpeg mtcnn VGG-Face euclidean` OR `python FaceVerifier.py ./dataset/kalam/1.jpg ./dataset/kalam/hi.jpeg` as by default, "retinaface" is used as detector and "ArcFace" as recognizer and "euclidean" as distance_metric

4) FaceDetector:- finds face region out of the whole input image of a person/identity

Syntax:- `python FaceDetector.py ./dataset/kalam/1.jpg face_detector_model`

Ex:- `python FaceDetector.py target_image_path mtcnn` OR `python FaceDetector.py ./dataset/kalam/1.jpg` as by default it has "retinaface"
