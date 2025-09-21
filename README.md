
# Face Recognition GUI with Siren and Email Alert

This project is a Python-based GUI application for face recognition in videos. When a specified face is detected in a video, the system alerts the user with a siren sound, a pop-up message, and optionally sends an email notification.

## Features
- Select an image of the person to detect
- Select a video file to scan
- Real-time face detection and recognition
- Alerts on detection:
  - Siren sound (`siren.mp3`)
  - Pop-up message
  - Email notification (configurable)

## Requirements
- Python 3.7+
- [DeepFace](https://github.com/serengil/deepface)
- OpenCV (`cv2`)
- Pillow (`PIL`)
- playsound

Install dependencies:
```bash
pip install deepface opencv-python pillow playsound
```

## Setup
1. **Clone or download this repository.**
2. **Place a siren sound file named `siren.mp3` in the project directory.**
3. **(Optional) Configure email alerts:**
   - Edit `gui_launcher_siren_email.py`.
   - In the `send_email_alert` function, set:
     - `sender` to your email address
     - `receiver` to the recipient's email address
     - `password` to your [app password](https://support.google.com/accounts/answer/185833?hl=en) (not your main password)

## How to Get a Gmail App Password
1. Enable 2-Step Verification on your Google account.
2. Go to [App Passwords](https://security.google.com/settings/security/apppasswords).
3. Generate a password for "Mail" and copy it.
4. Use this password in the script.

## Usage
1. Run the GUI:
   ```bash
   python gui_launcher_siren_email.py
   ```
2. Click **Select Person Image** and choose a photo of the person to detect.
3. Click **Select Video File** and choose a video to scan.
4. Click **Run Detection**.
5. If the person is detected:
   - A siren will sound
   - A pop-up message will appear
   - An email will be sent (if configured)

## Notes
- The detection uses DeepFace with the Facenet model.
- Make sure `siren.mp3` is present in the same directory as the script.
- For email alerts, use an app password for security.

## License
This project is for educational purposes.
