import mediapipe as mp
import cv2
import time

# Hand detection setup
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Face detection setup
faceCascade = cv2.CascadeClassifier('data_libs\\facedata.xml')

# Fps setup variables
pTime = 0
cTime = 0

# Videocapture
cap = cv2.VideoCapture(0)  

while True:           

    # capture things
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # fps show
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f'Fps:{int(fps)}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)


    # Show things:

    # Faces
    faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30), 
        )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(frame, 'Face', (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    # Hands

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Face detect IA - by @HotAndLonely", frame)
    if cv2.waitKey(1) == 27:
        break
