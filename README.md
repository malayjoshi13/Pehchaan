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

`python FaceFinder.py ./dataset/kalam/1.jpg mtcnn`
