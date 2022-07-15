import cv2
import utils.face_aligner as face_aligner 
from mtcnn import MTCNN


def build_model():
	face_detector = MTCNN()
	return face_detector



def detect_face(face_detector, img):

	resp = []

	detected_face = None
	img_region = [0, 0, img.shape[0], img.shape[1]]

	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	detections = face_detector.detect_faces(img_rgb) # "detect_faces" is function of mtcnn 

	if len(detections) > 0:
		for detection in detections:
			x, y, w, h = detection["box"]
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]
			img_region = [x, y, w, h]

			keypoints = detection["keypoints"]
			left_eye = keypoints["left_eye"]
			right_eye = keypoints["right_eye"]
			detected_face = face_aligner.align_face(detected_face, left_eye, right_eye)

			resp.append((detected_face, img_region))

	return resp