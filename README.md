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

**Ex 1**: `python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset mtcnn VGG-Face euclidean` --> here "mtcnn" is used as face detector and "VGGFace" as face recognizer and "euclidean" as distance_metric

**Ex 2**: `python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset` --> here by default, "retinaface" is used as face detector, "ArcFace" as face recognizer and "euclidean" as the distance metric

```Note: Rather than fine-tuning the pre-trained face recognition models, I choose to calculate the distance between feature vectors of input image and images in database. This is because having a small database in this use-case, fine-tuning didn't give the expected results (shown by results between two approaches below).```

## 3.2) SimilarFaceFinder:-
**Application**: finding all database images which look similar to the face in the input image.

**Under the hood**: `SimilarFaceFinder` algorithm applies the `FaceVerifier` algorithm in 1:N format between the input image and each database image to find the database image and input image having a distance (between their corresponding feature vectors) less than the threshold value. Whichever database image satisfies this condition is considered similar to the input image.

Syntax:- `python SimilarFaceFinder.py target_image_path database_path face_detector_model face_recognizer_model distance_metric`

**Ex 1**: `python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset mtcnn VGG-Face euclidean` --> here "mtcnn" is used as face detector and "VGGFace" as face recognizer and "euclidean" as distance_metric

**Ex 2**: `python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset` --> here by default, "retinaface" is used as face detector, "ArcFace" as face recognizer and "euclidean" as distance metric

## 4) Options to choose from

4.1) For `face_detector_model`:- opencvhaar, opencvdnn, dlib, mtcnn, retinaface, mediapipe

4.2) For `face_recognizer_model`:- VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, SFace

## 5) Use cases

5.1) auto-tagging individual(s) out of image(s). Developing a database with different folders having images of different peoples. Now we will use this database to auto-tag people in photos like image A has Ram, Shyam and Sita. But how? For suppose this image A enters into Pehchaan, then first face(s) are detected in image A --> then face alignment --> then face embedding --> then comparison with face embeddings of database images using SVM or cosine distance.

5.2) tracking shoppers in cashierless shops/stores

5.3) tracking movement of a specific criminal/suspect/individual

5.4) smart attendance/check-in/entry in event/office/college/home/school

5.5) biometric/phone/system authentication, etc

## 6) Working on

6.1) Understanding and conducting comparative analysis between algos in `algos.md` file

6.2) adding varying illumination robustness on top of face detection and recognition/identification pipeline by training on NIR images dataset (like https://www4.comp.polyu.edu.hk/~csajaykr/IITD/FaceIR.htm)
 
6.3) adding varying poses and face orientations robustness to existing pipeline by use of face alignment algorithm or by use of GANs for transfering features of posed input face to a straight reference face 

6.4) adding face-occlusion (face mask, veil, etc) robustness to existing pipeline by discriminative learning based on eyes feature vector (like https://scholar.google.co.in/citations?view_op=view_citation&hl=en&user=4crRvSMAAAAJ&citation_for_view=4crRvSMAAAAJ:Tyk-4Ss8FVUC)

6.5) image quality enhancement to make pipeline independent on camera quality (like https://publications.iitm.ac.in/publication/dp-gan-dual-pathway-generative-adversarial-network-for-face, https://scholar.google.co.in/citations?user=AZxz14AAAAAJ&hl=en, https://scholar.google.com/citations?user=JC528xwAAAAJ&hl=en, https://scholar.google.co.in/citations?view_op=view_citation&hl=en&user=JBAv0d0AAAAJ&citation_for_view=JBAv0d0AAAAJ:0EnyYjriUFMC,  https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html)

6.6) making pipeline more scalable by performing comparison between feature vectors of input and database images in 1 image : N groups format and not in 1 image : 1 image format. This will save extra time spent by not comparing input image to database images of a particular group whose centroid feature vector is at a very large distance. 

6.7) restructuring model structure to include required Fully Connected layers which have demonstrated better understanding of minute differences between face features. Also including required cost functions that can help in discriminative learning by increasing difference between non-similar classes.

