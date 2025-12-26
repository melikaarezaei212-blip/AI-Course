#import necessary libraries
import cv2
import numpy as np


#load pre-trained classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

# function to detect eye color
def detect_eye_color(eye_region):
    hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)

    # Focus on iris center
    h, w = eye_region.shape[:2]
    iris_roi = hsv[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]

    # Masks
    # Merge black into brown
    brown_mask = cv2.inRange(iris_roi, np.array([0, 0, 0]), np.array([25, 255, 200]))
    blue_mask = cv2.inRange(iris_roi, np.array([90, 30, 50]), np.array([140, 255, 255]))
    # Wider green range, lower S threshold
    green_mask = cv2.inRange(iris_roi, np.array([25, 25, 50]), np.array([95, 255, 255]))

    colors = {
        'Brown': np.sum(brown_mask),
        'Blue': np.sum(blue_mask),
        'Green': np.sum(green_mask)
    }

    detected_color = max(colors, key=colors.get)
    # Calculate confidence
    confidence = colors[detected_color] / (iris_roi.shape[0] * iris_roi.shape[1] * 255)

    return detected_color if confidence > 0.05 else None


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(80, 80))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        area_ratio = face_area / frame_area

        # Determine confidence based on area ratio and width
        if area_ratio > 0.15 and w > 150:
            confidence = 100
        elif area_ratio > 0.1:
            confidence = 90 + min(5, int((area_ratio - 0.1) * 100))
        elif area_ratio > 0.05:
            confidence = 80 + min(10, int((area_ratio - 0.05) * 200))
        else:
            confidence = 70 + min(10, int(area_ratio * 400))
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{confidence}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        # Analyze each detected eye for color
        eye_colors = []
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            if eye_roi.size > 0:
                eye_color = detect_eye_color(eye_roi)
                eye_colors.append(eye_color)

        if eye_colors:
            final_color = max(set(eye_colors), key=eye_colors.count)
            if final_color != "Unknown":
                cv2.putText(frame, f'Eye Color: {final_color}', (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),2)
    cv2.imshow('Face & Eye Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
