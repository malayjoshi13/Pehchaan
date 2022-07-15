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

## Execution

1) For identifying name of person (whose image is feeded as an input) out of the database (i.e. `dataset` folder)

Syntax:- `python FaceRecognizer.py target_image_path database_path face_detector_model face_recognizer_model distance_metric`

EX:- ```python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset mtcnn VGG-Face euclidean```

                                    OR
 
      ```python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset``` as by default, "retinaface" is used as detector and "ArcFace" as recognizer and    
       "euclidean" as distance metric
