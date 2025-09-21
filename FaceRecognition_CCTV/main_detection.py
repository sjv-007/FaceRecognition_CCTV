
import cv2
import numpy as np
from deepface import DeepFace
import os

THRESHOLD = 0.7
FRAME_SKIP = 3
OUTPUT_VIDEO = "output_detected.avi"

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def run_face_detection(input_image, video_source):
    input_embedding = DeepFace.represent(input_image, model_name="Facenet", enforce_detection=True)[0]["embedding"]

    cap = cv2.VideoCapture(video_source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frame_count = 0
    person_found = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_img)

            try:
                emb = DeepFace.represent(temp_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                sim = cosine_similarity(input_embedding, emb)
                label = f"Match: {sim:.2f}" if sim > THRESHOLD else "No Match"
                color = (255, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if sim > THRESHOLD:
                    person_found = True
            except:
                continue

        out.write(frame)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return person_found
