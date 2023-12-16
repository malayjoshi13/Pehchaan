# This module check if any new image is added or not in database.
# This checking helps to see if representation file need to be updated or not.

import os

# we only check if any new image is added and dont check if any image is removed, 
# because if any image would be added then whole representation file will be updated and we don't need to remove representations of image which is removed
def check_image_added(path):
    answer = False # initiallly assume no new image is added in database

    # if tracker file is present, then first analyse it
    if os.path.isfile(path+'/files_tracker.txt') == True:

        # list of already existing images in database
        with open(path+'/files_tracker.txt') as f:
            lines = f.readline()

        # checking if any new image is added in database
        for file in os.listdir(path):
            file = path+"/"+file
            if ('.jpg' in file.lower()) or ('.png' in file.lower()) or ('.jpeg' in file.lower()):
                if file not in lines:
                    print("new images detected in " + path+ " location. Recreating representations files")
                    # if found some new image in database, then set variable "answer" as True so that representation file could be updated
                    answer = True

    # if tracker file don't exist, it clearly means that its the first time this script is running, so representation file need to be created here
    else:
        answer = True

    return answer