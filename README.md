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

4.1) For `face_detector_model`:- OpenCVHaar, OpenCVDNN, Dlib, MTCNN, RetinaFace, MediaPipe

Weights for the face detector model get automatically downloaded from the internet; nothing is required from our end.

4.2) For face recognition:- <br>
a) VGGFace --> <br>
b) OpenFace --> https://cmusatyalab.github.io/openface/ | It's an unofficial Pytorch implementation of https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf developed at Google. <br>
c) Facenet --> https://github.com/davidsandberg/facenet | It's an unofficial Tensorflow implementation of https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf developed at Google and took reference from OpenFace library. <br>
d) DeepFace --> https://github.com/swghosh/DeepFace | It's an unofficial Tensorflow implementation of https://research.facebook.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/ developed at Meta. <br>
`Note`: Don't be confused between DeepFace model developed at Meta (mentioned above) and deepface library (with wrappers of many face recognition and detection models) developed by Serengil. <br>
e) DeepID --> https://github.com/kamwoh/face_recognition | It's an unofficial implementation of https://dl.acm.org/doi/10.1109/CVPR.2014.244 (DeepID 1 model). It has more variants like DeepID 2, DeepID 2+, DeepID 3. <br> 
f) ArcFace --> https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch | Official implementation of https://arxiv.org/abs/1801.07698. The same authors have also developed RetinaFace model and developed (insightface)[https://github.com/deepinsight/insightface] library for face detection and recognition. <br>
g) SFace --> <br>



## 5) Use cases

5.1) auto-tagging individual(s) out of image(s). Developing a database with different folders having images of different peoples. Now we will use this database to auto-tag people in photos like image A has Ram, Shyam and Sita. But how? For suppose this image A enters into Pehchaan, then first face(s) are detected in image A --> then face alignment --> then face embedding --> then comparison with face embeddings of database images using SVM or cosine distance.

5.2) tracking shoppers in cashierless shops/stores

5.3) tracking movement of a specific criminal/suspect/individual

5.4) smart attendance/check-in/entry in event/office/college/home/school

5.5) biometric/phone/system authentication, etc

## 6) Working on

6.1) Understanding each algo (like arcface, vggface, etc) in `algos.md` file and conducting comparative analysis between them by run them in different pairs to see which detector-recog pair works the best

also see how fine-tuninng face recog models create better reprsentations

extend to another usecase of locating different positions where a person (missing/thief) was using his/her single image. Thus save time to manually view whole CCTV footage

6.2) adding varying illumination robustness on top of face detection and recognition/identification pipeline by training on NIR images dataset (like https://www4.comp.polyu.edu.hk/~csajaykr/IITD/FaceIR.htm)
 
6.3) adding varying poses and face orientations robustness to existing pipeline by use of face alignment algorithm or by use of GANs for transfering features of posed input face to a straight reference face 

6.4) adding face-occlusion (face mask, veil, etc) robustness to existing pipeline by discriminative learning based on eyes feature vector (like https://scholar.google.co.in/citations?view_op=view_citation&hl=en&user=4crRvSMAAAAJ&citation_for_view=4crRvSMAAAAJ:Tyk-4Ss8FVUC)

6.5) image quality enhancement to make pipeline independent on camera quality (like https://publications.iitm.ac.in/publication/dp-gan-dual-pathway-generative-adversarial-network-for-face, https://scholar.google.co.in/citations?user=AZxz14AAAAAJ&hl=en, https://scholar.google.com/citations?user=JC528xwAAAAJ&hl=en, https://scholar.google.co.in/citations?view_op=view_citation&hl=en&user=JBAv0d0AAAAJ&citation_for_view=JBAv0d0AAAAJ:0EnyYjriUFMC,  https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html)

6.6) making pipeline more scalable by performing comparison between feature vectors of input and database images in 1 image : N groups format and not in 1 image : 1 image format. This will save extra time spent by not comparing input image to database images of a particular group whose centroid feature vector is at a very large distance. 

6.7) restructuring model structure to include required Fully Connected layers which have demonstrated better understanding of minute differences between face features. Also including required cost functions that can help in discriminative learning by increasing difference between non-similar classes.

