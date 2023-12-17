from retinaface import RetinaFace
from utils import face_aligner 
import cv2

def build_model():
	face_detector = RetinaFace
	return face_detector


def detect_face(face_detector, img):
    resp = []

    detected_face = None
    img_region = [0, 0, img.shape[0], img.shape[1]]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = face_detector.detect_faces(img_rgb) # "detect_faces" is function of mtcnn 
    
    if len(detections) > 0:
        x, y, w, h = detections["face_1"]["facial_area"]
        detected_face = img[int(y):int(h), int(x):int(w)]
        img_region = [x, y, w, h]

        keypoints = detections["face_1"]["landmarks"]
        left_eye = keypoints["left_eye"]
        right_eye = keypoints["right_eye"]
        detected_face = face_aligner.align_face(detected_face, left_eye, right_eye)

        resp.append((detected_face, img_region))

    return resp