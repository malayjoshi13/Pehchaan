[1] FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR 2015 [Task](face recognition) [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html) [arXiv](https://arxiv.org/abs/1503.03832) [Code](this code is more like use of facenet for an end-to-end pipeline, and is inspired by OpenFace's pipeline structure https://github.com/davidsandberg/facenet) [article](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/, https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/) 

[2] OpenFace: An open source facial behavior analysis toolkit [Task](face recognition & detection) [Paper](https://ieeexplore.ieee.org/abstract/document/7477553?casa_token=jC0fbSjjguAAAAAA:y0qXivGgDR_X9aygYQRgNn9Wln6k9N8leOfkoHFl9nP32Unai00Z14_d_gsJSQHHhvQwXKVP1g)  [Code](http://cmusatyalab.github.io/openface/, https://github.com/aakashjhawar/face-recognition-using-deep-learning) [Note] (OpenFace uses FaceNet for recognition work and dlib/opencv for detection) [Article](written by author of face_recognition library https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78#.ds8i8oic9, https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/)

[3] DeepFace Recognition [Task](face recognition & detection) [Paper](just paper is present, no code. Will use for theortical purpose)(https://ora.ox.ac.uk/objects/uuid:a5f2e93f-2768-45bb-8508-74747f85cad1/download_file?file_format=pdf&safe_filename=parkhi15.pdf&type_of_work=Confer) [Note](It uses Dlib at its base)

[4] LightFace: A Hybrid Deep Face Recognition Framework [Task](face recognition & detection) [Paper](https://ieeexplore.ieee.org/document/92598020) [Code](which was preesent above, https://github.com/serengil/deepface) [Note](I simplified this code so that it can be used for adding more libraries for our research work https://github.com/malayjoshi13/Pehchaan)

[5] Face Recognition [Task](face recognition & detection) [Code](https://github.com/ageitgey/face_recognition#face-recognition) [Note](no research backing, but amazing code based on Dlib)

[6] OpenCV a) Haarcascade [Task](face detection) [Article](https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81 , https://pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/, https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/) [Note](Pros: Very fast, capable of running in super real-time
Low computational requirements — can easily be run on embedded, resource-constrained devices such as the Raspberry Pi (RPi), NVIDIA Jetson Nano, and Google Coral, Small model size (just over 400KB; for reference, most deep neural networks will be anywhere between 20-200MB).

Cons: Highly prone to false-positive detections, Typically requires manual tuning to the detectMultiScale function, Not anywhere near as accurate as its HOG + Linear SVM and deep learning-based face detection counterparts

My recommendation: Use Haar cascades when speed is your primary concern, and you’re willing to sacrifice some accuracy to obtain real-time performance.)

b) DNN [Task](face detection) [Article](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/, https://github.com/aakashjhawar/face-recognition-using-deep-learning) [Note](it is based on a Single Shot Detector (SSD) with a small ResNet backbone, allowing it to be both accurate and fast.

Pros: Accurate face detector, Utilizes modern deep learning algorithms, No parameter tuning required, Can run in real-time on modern laptops and desktops
Model is reasonably sized (just over 10MB), Relies on OpenCV’s cv2.dnn module, Can be made faster on embedded devices by using OpenVINO and the Movidius NCS

Cons: More accurate than Haar cascades and HOG + Linear SVM, but not as accurate as dlib’s CNN MMOD face detector, May have unconscious biases in the training set — may not detect darker-skinned people as accurately as lighter-skinned people

My recommendation: OpenCV’s deep learning face detector is your best “all-around” detector. It’s very simple to use, doesn’t require additional libraries, and relies on OpenCV’s cv2.dnn module, which is baked into the OpenCV library. Perhaps the biggest downside of this model is that I’ve found that the face detections on darker-skinned people aren’t as accurate as lighter-skinned people. That’s not necessarily a problem with the model itself but rather the data it was trained on — to remedy that problem, I suggest training/fine-tune the face detector on a more diverse set of ethnicities.)

[7] Dlib [Task](face detection and reconition) [Article](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/, https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/, https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/) [Note] (Dlib HOG Pros: More accurate than Haar cascades, More stable detection than Haar cascades (i.e., fewer parameters to tune), Expertly implemented by dlib creator and maintainer, Davis King, Extremely well documented, both in terms of the dlib implementation and the HOG + Linear SVM framework in the computer vision literature

Cons: Only works on frontal views of the face — profile faces will not be detected as the HOG descriptor does not tolerate changes in rotation or viewing angle well, Requires an additional library (dlib) be installed — not necessarily a problem per se, but if you’re using just OpenCV, then you may find adding another library into the mix cumbersome, Not as accurate as deep learning-based face detectors
For the accuracy, it’s actually quite computationally expensive due to image pyramid construction, sliding windows, and computing HOG features at every stop of the window

My recommendation: HOG + Linear SVM is a classic object detection algorithm that every computer vision practitioner should understand. That said, for the accuracy HOG + Linear SVM gives you, the algorithm itself is quite slow, especially when you compare it to OpenCV’s SSD face detector. I tend to use HOG + Linear SVM in places where Haar cascades aren’t accurate enough, but I cannot commit to using OpenCV’s deep learning face detector.)

(Dlib CNN Pros: Davis King, the creator of dlib, trained a CNN face detector based on his work on max-margin object detection, Incredibly accurate face detector, Small model size (under 1MB), Expertly implemented and documented

Cons: Requires an additional library (dlib) be installed, Code is more verbose — end-user must take care to convert and trim bounding box coordinates if using OpenCV, Cannot run in real-time without GPU acceleration.

Recommendation: I tend to use dlib’s MMOD CNN face detector when batch processing face detection offline, meaning that I can set up my script and let it run in batch mode without worrying about real-time performance. In fact, when I build training sets for face recognition, I often use dlib’s CNN face detector to detect faces before training the face recognizer itself. When I’m ready to deploy my face recognition model, I’ll often swap out dlib’s CNN face detector for a more computationally efficient one that can run in real-time (e.g., OpenCV’s CNN face detector). 
The only place I tend not to use dlib’s CNN face detector is when I’m using embedded devices. This model will not run in real-time on embedded devices, and it’s out-of-the-box compatible with embedded device accelerators like the Movidius NCS.)

[8] MTCNN [Task](face detection) [Paper](https://arxiv.org/abs/1604.02878) [Code](https://github.com/davidsandberg/facenet/tree/master/src/align, https://github.com/ipazc/mtcnn) [Article](https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/)

[9] Mediapipe [Task](face detection) [Code](https://google.github.io/mediapipe/solutions/face_detection), [Article] (https://towardsdatascience.com/write-a-few-lines-of-code-and-detect-faces-draw-landmarks-from-complex-images-mediapipe-932f07566d11)
 
[10] Retinaface [Task](face detection) [Paper](https://arxiv.org/pdf/1905.00641.pdf) [Code](https://github.com/StanislasBertrand/RetinaFace-tf2 converted C version of RetinaFace into python. Then author of DeepFace (https://github.com/serengil/deepface), copied it and simplified the python version https://www.youtube.com/watch?v=Wm1DucuQk70 and https://github.com/serengil/retinaface) 

[11]  SSH: Single Stage Headless Face Detector [Task](face detection) [Article+code+paper] (https://medium.com/analytics-vidhya/exploring-other-face-detection-approaches-part-2-ssh-7c85179cd98d)

[12] PCN: Progressive Calibration Networks [Task](face detection) [Article+code+paper](https://medium.com/analytics-vidhya/exploring-other-face-detection-approaches-part-3-pcn-395d3b07d62a)

[13] Tiny Face Detector [Task](face detection) [Article+code+paper](https://medium.com/analytics-vidhya/exploring-other-face-detection-approaches-part-4-tiny-face-684c8cba5b01)

[14] For face alignment [https://arxiv.org/pdf/1703.07332.pdf, https://github.com/1adrianb/face-alignment] and [https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/] and [https://rajathithanrajasekar.medium.com/opencv-series-6-face-alignment-4a2779b6070d]

[15] VGG-Face [Task](face recognition) [Article](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)

[16] ArcFace [Task](face recognition) [Article](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/)

[17] Facebook DeepFace [Task](face recognition) [Article](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/)

[18] DeepID [Task](face recognition) [Article](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/)

[19] Histogram of Oriented Gradients [Task](face detection)
