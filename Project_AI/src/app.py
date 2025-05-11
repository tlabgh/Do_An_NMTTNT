import streamlit as st
from datetime import date, timedelta
import cv2
from PIL import Image
import os
import tempfile
import face_recognition
import numpy as np
import json
from data_utils import load_json, save_json
from image_utils import save_captured_image
from face_recognition_utils import load_known_faces, compare_face, find_existing_face

# ===== Cấu hình đường dẫn =====

db_user_path = r"E:\Project_AI\src\user_data.json"
faces_db_path = r"E:\Project_AI\src\face_data.json"
photo_dir = "E:\Project_AI\image_storage"
temp_path = "E:/Project_AI/src/temp.jpg"

# ===== Hàm hỗ trợ =====
def load_json_database(db_path):
    try:
        with open(db_path, "r") as f:
            db = json.load(f)
        encodings = [np.array(item.get("encoding")) for item in db if item.get("encoding")]
        names = [item.get("name") for item in db]
        return encodings, names
    except FileNotFoundError:
        return [], []

# Hàm thêm khuôn mặt vào face_data.json
def add_face_to_db(image_path, name, db_path=faces_db_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        st.error("Không tìm thấy khuôn mặt trong ảnh!")
        return
    encoding = encodings[0].tolist()
    try:
        with open(db_path, "r") as f:
            db = json.load(f)
    except FileNotFoundError:
        db = []
    db.append({"name": name, "encoding": encoding})
    with open(db_path, "w") as f:
        json.dump(db, f)
    st.success(f"Đã thêm {name} vào database.")

# ===== Giao diện chính =====
# Logo và tên học viện
t_col1, t_col2 = st.columns([1,5])
with t_col1:
    st.image("E:/Project_AI/src_images/ptit.png", width=80)
with t_col2:
    st.markdown(
        "<h3 style='margin:0;'>HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG</h3>",
        unsafe_allow_html=True
    )

# Sidebar: chọn chức năng
st.sidebar.title("Face Recognition System")
modes = [
    "Đăng ký người dùng",
    "Nhận diện bằng webcam",
    "Nhận diện bằng ảnh & video"
]
mode = st.sidebar.radio("Chức năng:", modes)

# Nạp dữ liệu khuôn mặt đã đăng ký và face_data.json
known_encs_file, known_names_file = load_json_database(faces_db_path)

# ===== 1. Đăng ký người dùng =====
if mode == modes[0]:
    # ... (các input như cũ)
    st.header("Đăng ký người dùng mới")
    name = st.text_input("Họ tên:")
    today = date.today()
    min_dob = today - timedelta(days=365*100)
    dob = st.date_input("Ngày sinh:", value=today, min_value=min_dob, max_value=today)
    hometown = st.text_input("Quê quán:")
    img_data = st.camera_input("Chụp khuôn mặt:")
    if st.button("Lưu thông tin"):
        if not name or not img_data:
            st.error("Vui lòng nhập đủ thông tin và chụp ảnh.")
        else:
            # Lưu tạm file và chuyển sang RGB
            with open(temp_path, 'wb') as f:
                f.write(img_data.getvalue())
            bgr = cv2.imread(temp_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Gọi hàm kiểm tra trùng
            existing_uid = find_existing_face(rgb)
            if existing_uid:
                st.error(f"Người dùng đã tồn tại trong hệ thống (UID={existing_uid}).")
                os.remove(temp_path)
            else:
                # Nếu chưa có, tạo UID mới và move ảnh vào kho
                users = load_json(db_user_path) #load users lên
                numbers = [int(u["id"][2:]) for u in users] # lấy id các user sau "MS"
                max_num = max(numbers, default=0) #Tìm max
                uid = f"MS{max_num + 1:03d}"
                fname = f"{uid}.jpg"
                os.rename(temp_path, os.path.join(photo_dir, fname))

                # Lưu record vào user_data.json
                rec = {
                    "id": uid,
                    "name": name.strip(),
                    "dob": dob.isoformat(),
                    "hometown": hometown.strip(),
                    "image": fname
                }
                users.append(rec)
                save_json(users, db_user_path)

                st.success(f"Đăng ký thành công! UserID: {uid}")


# ===== 2. Nhận diện bằng webcam =====
elif mode == modes[1]:
    st.header("Nhận diện khuôn mặt bằng webcam")
    img_data = st.camera_input("Quét khuôn mặt:")
    if st.button("Xác nhận"):
        if not img_data:
            st.error("Vui lòng quét khuôn mặt.")
        else:
            with open(temp_path, 'wb') as f:
                f.write(img_data.getvalue())
            bgr = cv2.imread(temp_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            known_encs_webcam, known_ids_webcam = load_known_faces()
            if len(known_encs_webcam)==0 and len(known_ids_webcam)==0:
                st.error("Không tìm thấy thông tin người dùng.")
            else:
                mid = compare_face(known_encs_webcam, known_ids_webcam, rgb)
                #MS0001
                print(mid)
                if mid:
                    users = load_json(db_user_path)
                    user = next((u for u in users if u.get('id') == mid), None)
                    print(user)
                    # [{'id': 'MS0001', 'name': 'Trần Công Hậu', 'dob': '2025-05-11',
                    # 'hometown': 'Ấp đức ngãi 2, xã đức lập thượng, huyện đức hòa, tỉnh long an',
                    # 'image': 'MS0001.jpg'}]
                    if user:
                        st.write(f"**Họ tên:** {user['name']}")
                        st.write(f"**Ngày sinh:** {user['dob']}")
                        st.write(f"**UserID:** {user['id']}")
                        st.write(f"**Quê quán:** {user['hometown']}")
                        img_path = os.path.join(photo_dir, user['image'])
                        st.image(Image.open(img_path), caption="Ảnh lưu trữ")
                    else:
                        st.error("Không tìm thấy thông tin người dùng.")
                else:
                    st.error("Khuôn mặt không khớp với bất kỳ ai.")
                os.remove(temp_path)

# ===== 3. Nhận diện bằng ảnh & video =====
else:
    st.header("Nhận diện khuôn mặt bằng ảnh & video")

    # -- 3.0 Thêm khuôn mặt mới vào database --
    st.subheader("Thêm khuôn mặt mới vào database")
    with st.form("add_face_form"):
        new_name = st.text_input("Tên người:")
        new_image = st.file_uploader("Chọn ảnh khuôn mặt", type=["jpg","jpeg","png"], key="add_face_3")
        if st.form_submit_button("Thêm vào database"):
            if not new_name or not new_image:
                st.error("Vui lòng nhập tên và chọn ảnh.")
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.write(new_image.getbuffer())
                tmp.close()
                add_face_to_db(tmp.name, new_name)
                os.remove(tmp.name)

    # -- 3.1 Nhận diện từ ảnh tĩnh --
    st.subheader("Nhận diện từ ảnh")
    uploaded_file = st.file_uploader("Chọn ảnh để nhận diện", type=["jpg","jpeg","png"], key="img_rec")
    if uploaded_file:
        data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(data, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        for (top,right,bottom,left), enc in zip(locs, encs):
            matches = face_recognition.compare_faces(known_encs_file, enc, tolerance=0.65)
            name = "Unknown"
            if True in matches:
                dists = face_recognition.face_distance(known_encs_file, enc)
                idx = np.argmin(dists)
                if matches[idx]: name = known_names_file[idx]
            cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(img, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Kết quả nhận diện ảnh", use_container_width=True)

    # -- 3.2 Nhận diện từ video --
    st.subheader("Nhận diện từ video")
    video_file = st.file_uploader("Chọn video để nhận diện", type=["mp4","avi","mov"], key="vid_rec")
    if video_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
        tmp.write(video_file.read())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        stframe = st.empty()
        count = 0
        while cap.isOpened() and count < 100:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            for (top,right,bottom,left), enc in zip(locs, encs):
                matches = face_recognition.compare_faces(known_encs_file, enc, tolerance=0.65)
                name = "Unknown"
                if True in matches:
                    dists = face_recognition.face_distance(known_encs_file, enc)
                    idx = np.argmin(dists)
                    if matches[idx]: name = known_names_file[idx]
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            stframe.image(frame, channels="BGR")
            count += 1
        cap.release()