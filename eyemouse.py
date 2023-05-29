#Not working properly in PyCharm. Try in Visual Studio Code (Codespaces)
import cv2
import dlib
import math
import pyautogui

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye aspect ratio (EAR) threshold to detect eye blinks
EAR_THRESH = 0.2

# Define consecutive frames for which eye blink must be detected to trigger an action
EAR_CONSEC_FRAMES = 3

COUNTER = 0
BLINKED = False

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract the left and right eye landmarks
        left_eye = [landmarks.part(36).x, landmarks.part(36).y,
                    landmarks.part(37).x, landmarks.part(37).y,
                    landmarks.part(38).x, landmarks.part(38).y,
                    landmarks.part(39).x, landmarks.part(39).y,
                    landmarks.part(40).x, landmarks.part(40).y,
                    landmarks.part(41).x, landmarks.part(41).y]
        right_eye = [landmarks.part(42).x, landmarks.part(42).y,
                     landmarks.part(43).x, landmarks.part(43).y,
                     landmarks.part(44).x, landmarks.part(44).y,
                     landmarks.part(45).x, landmarks.part(45).y,
                     landmarks.part(46).x, landmarks.part(46).y,
                     landmarks.part(47).x, landmarks.part(47).y]

        # Calculate eye aspect ratio (EAR) for each eye
        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)

        # Calculate the average EAR for both eyes
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < EAR_THRESH:
            COUNTER += 1

            if COUNTER >= EAR_CONSEC_FRAMES:
                if not BLINKED:
                    # Perform eye blink action
                    pyautogui.click(button='left')
                BLINKED = True
        else:
            COUNTER = 0
            BLINKED = False

        # Draw a rectangle around the face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Draw circles around the left and right eye landmarks
        for i in range(36, 42):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 0, 255), -1)
        for i in range(42, 48):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 0, 255), -1)

    # Show the frame
    cv2.imshow("Eye Blink Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()