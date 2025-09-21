import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import cv2
from main_detection import run_face_detection, cosine_similarity
from deepface import DeepFace

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("900x600")
        self.root.configure(bg="white")

        self.image_path = None
        self.video_path = None
        self.cap = None
        self.running = False

        tk.Label(root, text="Face Recognition System", font=("Helvetica", 20), bg="white").pack(pady=10)

        btn_frame = tk.Frame(root, bg="white")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Select Person Image", command=self.select_image, font=("Helvetica", 12)).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Select Video File", command=self.select_video, font=("Helvetica", 12)).grid(row=0, column=1, padx=10)

        self.run_button = tk.Button(btn_frame, text="Run Detection", command=self.run_detection, font=("Helvetica", 12), state=tk.DISABLED)
        self.run_button.grid(row=0, column=2, padx=10)

        self.status_label = tk.Label(root, text="", font=("Helvetica", 12), bg="white")
        self.status_label.pack(pady=5)

        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(pady=10)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp")])
        self.check_ready()

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        self.check_ready()

    def check_ready(self):
        if self.image_path and self.video_path:
            self.run_button.config(state=tk.NORMAL)

    def run_detection(self):
        self.status_label.config(text="Running detection...")
        self.running = True
        threading.Thread(target=self.detect_thread).start()

    def detect_thread(self):
        input_embedding = DeepFace.represent(self.image_path, model_name="Facenet", enforce_detection=True)[0]["embedding"]
        self.cap = cv2.VideoCapture(self.video_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        THRESHOLD = 0.7
        FRAME_SKIP = 3
        frame_count = 0

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
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
                except:
                    continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.cap.release()
        self.status_label.config(text="Detection completed.")

root = tk.Tk()
app = FaceRecognitionGUI(root)
root.mainloop()
