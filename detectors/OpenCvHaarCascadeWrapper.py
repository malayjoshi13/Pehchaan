import cv2
import utils.face_aligner as face_aligner 
import os

def build_cascade(model_name):
    if model_name == 'haarcascade':
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return face_detector

    elif model_name == 'haarcascade_eye':
        eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        return eye_detector



def build_model():
	detector ={}
	detector["face_detector"] = build_cascade('haarcascade')
	detector["eye_detector"] = build_cascade('haarcascade_eye')
	return detector



#------------------------------------------------------------


def find_eyes_for_opencv(eye_detector, img):

	detected_face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #eye detector expects gray scale image

	#eyes = eye_detector.detectMultiScale(detected_face_gray, 1.3, 5)
	eyes = eye_detector.detectMultiScale(detected_face_gray, 1.1, 10)

	#----------------------------------------------------------------

	#opencv eye detectin module is not strong. it might find more than 2 eyes!
	#besides, it returns eyes with different order in each call (issue 435)
	#this is an important issue because opencv is the default detector and ssd also uses this
	#find the largest 2 eye. Thanks to @thelostpeace

	eyes = sorted(eyes, key = lambda v: abs((v[0] - v[2]) * (v[1] - v[3])), reverse=True)

	#----------------------------------------------------------------
	
	if len(eyes) >= 2:

		#decide left and right eye

		eye_1 = eyes[0]; eye_2 = eyes[1]

		if eye_1[0] < eye_2[0]:
			left_eye = eye_1; right_eye = eye_2
		else:
			left_eye = eye_2; right_eye = eye_1

		#-----------------------
		#find center of eyes
		left_eye = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
		right_eye = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
		
	return (left_eye, right_eye) 



def detect_face(detector, img):

	resp = []

	detected_face = None
	img_region = [0, 0, img.shape[0], img.shape[1]]

	faces = []
	try:
		faces = detector["face_detector"].detectMultiScale(img, 1.1, 10)
	except:
		pass

	if len(faces) > 0:

		for x,y,w,h in faces:
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]
			left_eye, right_eye = find_eyes_for_opencv(detector["eye_detector"], detected_face)
			detected_face = face_aligner.align_face(detected_face, left_eye, right_eye)
			img_region = [x, y, w, h]
			resp.append((detected_face, img_region))

	return resp
