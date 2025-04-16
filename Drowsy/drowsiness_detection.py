# drowsiness.py
import cv2
import dlib
import time
from scipy.spatial import distance
from imutils import face_utils
import pygame

def run_drowsiness_detection():
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.mp3")

    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 48

    COUNTER = 0
    WARNING_COUNTER = 0
    WARNING_START_TIME = None

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()

                    if WARNING_START_TIME is None:
                        WARNING_START_TIME = time.time()
                        WARNING_COUNTER += 1
            else:
                COUNTER = 0
                pygame.mixer.music.stop()
                WARNING_START_TIME = None

            if WARNING_COUNTER >= 3:
                elapsed = time.time() - WARNING_START_TIME if WARNING_START_TIME else 0
                if elapsed < 120:
                    cv2.putText(frame, "Please take a rest!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    WARNING_COUNTER = 0
                    WARNING_START_TIME = None

            for eye in [leftEye, rightEye]:
                hull = cv2.convexHull(eye)
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
