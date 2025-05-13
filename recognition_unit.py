import os
import face_recognition
import numpy as np

encoding_dir = '/home/madhavr/Desktop/biometric_recognition/recog_system/encoding_faces'

def recognize_from_group(group_image_path):
    """
    Input: group photo path
    Output: prints roll numbers of matched faces
    """
    # Load known encodings
    known_encodings = []
    roll_numbers = []

    for file in os.listdir(encoding_dir):
        if file.endswith('.npy'):
            roll_no = os.path.splitext(file)[0]
            encoding = np.load(os.path.join(encoding_dir, file))
            known_encodings.append(encoding)
            roll_numbers.append(roll_no)

    print(f"[INFO] Loaded {len(known_encodings)} encodings.")

    # Load group image
    group_image = face_recognition.load_image_file(group_image_path)
    group_encodings = face_recognition.face_encodings(group_image)

    if not group_encodings:
        print("[⚠] No faces detected in group image.")
        return

    print(f"[INFO] Detected {len(group_encodings)} face(s) in group photo.\n")

    # Compare each group face with known encodings
    for idx, group_encoding in enumerate(group_encodings):
        found_match = False
        for known_encoding, roll_no in zip(known_encodings, roll_numbers):
            match = face_recognition.compare_faces([known_encoding], group_encoding, tolerance=0.5)
            if match[0]:
                print(f"[✓] Match found: {roll_no} (Face #{idx + 1})")
                found_match = True
                break

        if not found_match:
            print(f"[✗] Face #{idx + 1}: No match found.")

recognize_from_group('/home/madhavr/Desktop/biometric_recognition/recog_system/WhatsApp Image 2025-05-13 at 2.30.19 PM.jpeg')
# https://github.com/madhavprapanna52/Face_detection_project.git