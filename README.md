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

4.1) For `face_detector_model`:- opencvhaar, opencvdnn, dlib, mtcnn, retinaface, mediapipe

4.2) For `face_recognizer_model`:- VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, SFace

## 5) Use cases

5.1) auto-tagging individual(s) out of image(s)

5.2) tracking shoppers in cashierless shops/stores

5.3) tracking movement of a specific criminal/suspect/individual

5.4) smart attendance/check-in/entry in event/office/college/home/school

5.5) biometric/phone/system authentication, etc

## 6) Working on

6.1) Understanding and conducting comparative analysis between following algos https://github.com/malayjoshi13/face_recognition_detection_research/blob/main/Research%20papers.md

6.2) adding varying illumination robustness on top of face detection and recognition/identification pipeline by training on NIR images dataset (like https://www4.comp.polyu.edu.hk/~csajaykr/IITD/FaceIR.htm)
 
6.3) adding varying poses and face orientations robustness to existing pipeline by use of face alignment algorithm or by use of GANs for transfering features of posed input face to a straight reference face 

6.4) adding face-occlusion (face mask, veil, etc) robustness to existing pipeline by discriminative learning based on eyes feature vector (like https://scholar.google.co.in/citations?view_op=view_citation&hl=en&user=4crRvSMAAAAJ&citation_for_view=4crRvSMAAAAJ:Tyk-4Ss8FVUC)

6.5) image quality enhancement to make pipeline independent on camera quality (like https://publications.iitm.ac.in/publication/dp-gan-dual-pathway-generative-adversarial-network-for-face, https://scholar.google.co.in/citations?user=AZxz14AAAAAJ&hl=en, https://scholar.google.com/citations?user=JC528xwAAAAJ&hl=en, https://scholar.google.co.in/citations?view_op=view_citation&hl=en&user=JBAv0d0AAAAJ&citation_for_view=JBAv0d0AAAAJ:0EnyYjriUFMC,  https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html)

6.6) making pipeline more scalable by performing comparison between feature vectors of input and database images in 1 image : N groups format and not in 1 image : 1 image format. This will save extra time spent by not comparing input image to database images of a particular group whose centroid feature vector is at a very large distance. 

6.7) restructuring model structure to include required Fully Connected layers which have demonstrated better understanding of minute differences between face features. Also including required cost functions that can help in discriminative learning by increasing difference between non-similar classes.

