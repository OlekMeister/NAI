"""
Smile Detection Program

Author: Aleksander Guzik

Description:
This program uses Haar cascades to detect smiles in real-time using the webcam.
Press the 'Esc' key to exit the program.

Requirements:
- opencv-python (install with: pip install opencv-python)

Usage:
- Run the script, and it will open a window displaying the webcam feed with smile detection.

Press 'Esc' to exit the program.
"""
import cv2

def smile_detection():
    # Initialize the classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the classifier for smile detection
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Smile detection
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                cv2.putText(frame, 'Smile detected!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Smile Detection', frame)

        # Exit the loop by pressing the 'Esc' key
        if cv2.waitKey(1) == 27:
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the smile detection function
smile_detection()
