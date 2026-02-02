import cv2
import mediapipe as mp
import time
import math
import winsound
import threading
import numpy as np


SYSTEM_START_TIME = time.time()
ALERT_ENABLE_DELAY = 5  # seconds


def play_siren(duration=2):
    def siren():
        end_time = time.time() + duration
        while time.time() < end_time:
            winsound.Beep(1000, 300)
    threading.Thread(target=siren, daemon=True).start()


def dist(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist(mouth[2], mouth[10])
    B = dist(mouth[4], mouth[8])
    C = dist(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def get_yaw_angle(landmarks, w, h):
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),     # Nose
        (landmarks[152].x * w, landmarks[152].y * h), # Chin
        (landmarks[33].x * w, landmarks[33].y * h),   # Left eye
        (landmarks[263].x * w, landmarks[263].y * h), # Right eye
        (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth
        (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    _, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[1]  # Yaw

mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
FaceLandmarker = mp_tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp_tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp_tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

face_landmarker = FaceLandmarker.create_from_options(options)


LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.60
DROWSY_TIME = 2
YAW_THRESHOLD = 25
HEAD_TURN_TIME = 2

eye_start_time = None
yawn_start_time = None
head_turn_start_time = None

prev_drowsy = 0
prev_yawn = 0
prev_head_turn = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = face_landmarker.detect_for_video(mp_image, timestamp)

    drowsy = 0
    yawn = 0
    head_turn = 0

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        left_eye, right_eye, mouth = [], [], []

        for i in LEFT_EYE:
            lm = landmarks[i]
            left_eye.append((int(lm.x * w), int(lm.y * h)))

        for i in RIGHT_EYE:
            lm = landmarks[i]
            right_eye.append((int(lm.x * w), int(lm.y * h)))

        for i in MOUTH:
            lm = landmarks[i]
            mouth.append((int(lm.x * w), int(lm.y * h)))

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        if ear < EAR_THRESHOLD:
            if eye_start_time is None:
                eye_start_time = time.time()
            elif time.time() - eye_start_time >= DROWSY_TIME:
                drowsy = 1
                cv2.putText(frame, "DROWSINESS ALERT!", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            eye_start_time = None

        mar = mouth_aspect_ratio(mouth)
        if mar > MAR_THRESHOLD:
            if yawn_start_time is None:
                yawn_start_time = time.time()
            elif time.time() - yawn_start_time > 1:
                yawn = 1
                cv2.putText(frame, "YAWNING DETECTED!", (40, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        else:
            yawn_start_time = None

        yaw_angle = get_yaw_angle(landmarks, w, h)
        cv2.putText(frame, f"Yaw: {int(yaw_angle)}°", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        if abs(yaw_angle) > YAW_THRESHOLD:
            if head_turn_start_time is None:
                head_turn_start_time = time.time()
            elif time.time() - head_turn_start_time >= HEAD_TURN_TIME:
                head_turn = 1
                cv2.putText(frame, "HEAD TURN ALERT!", (40, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        else:
            head_turn_start_time = None

        if (drowsy == 1 and prev_drowsy == 0) or \
           (yawn == 1 and prev_yawn == 0) or \
           (head_turn == 1 and prev_head_turn == 0):

            if time.time() - SYSTEM_START_TIME > ALERT_ENABLE_DELAY:
                play_siren()

        prev_drowsy = drowsy
        prev_yawn = yawn
        prev_head_turn = head_turn

    cv2.imshow("AI Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
