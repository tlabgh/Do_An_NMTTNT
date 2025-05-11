# face_recognition_utils.py
import os
import numpy as np
import cv2
import face_recognition

# Thư mục chứa ảnh mẫu tĩnh
PHOTO_DIR = "E:\Project_AI\image_storage"
THRESHOLD = 0.35


def load_known_faces():
    known_encodings, known_ids = [], []
    for fname in os.listdir(PHOTO_DIR):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        uid = os.path.splitext(fname)[0]
        path = os.path.join(PHOTO_DIR, fname)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            known_encodings.append(encs[0])
            known_ids.append(uid)
    return known_encodings, known_ids


def compare_face(known_encodings, known_ids, rgb_frame):
    encs = face_recognition.face_encodings(rgb_frame)
    if not encs:
        return None
    distances = face_recognition.face_distance(known_encodings, encs[0])
    idx = np.argmin(distances)
    return known_ids[idx] if distances[idx] < THRESHOLD else None
# Thêm hàm kiểm tra trùng
def find_existing_face(rgb_frame, threshold=THRESHOLD):
    """
    Trả về UID nếu khuôn mặt trong rgb_frame đã tồn tại,
    hoặc None nếu chưa có.
    """
    known_encs, known_ids = load_known_faces()
    # Nếu kho rỗng ko có dữ liệu thì được thêm vào
    if len(known_encs)==0 and len(known_ids)==0:
        return None
    # Nếu trả về 1 known_ids nghĩa là đã tồn tại không được thêm, nếu trả về None là được thêm
    return compare_face(known_encs, known_ids, rgb_frame)