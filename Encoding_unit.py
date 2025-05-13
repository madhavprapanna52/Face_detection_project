import os
import face_recognition
import numpy as np
from pathlib import Path

student_dir = '/home/madhavr/Desktop/biometric_recognition/recog_system/Data_set'
encoding_dir = '/home/madhavr/Desktop/biometric_recognition/recog_system/encoding_faces'

def encode_student_face(image_filename):
    """
    Input: student image file name (should exist in Data_set/)
    Output: saves a .npy file of face encoding in encoding_faces/
    """
    roll_no = Path(image_filename).stem
    image_path = os.path.join(student_dir, image_filename)

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print(f"[⚠] No face detected in {image_filename}")
        return

    encoding = encodings[0]
    output_path = os.path.join(encoding_dir, f"{roll_no}.npy")
    np.save(output_path, encoding)
    print(f"[✓] Encoding saved for {roll_no} at {output_path}")

encode_student_face('/home/madhavr/Desktop/biometric_recognition/recog_system/Data_set/24f3004979.jpeg')