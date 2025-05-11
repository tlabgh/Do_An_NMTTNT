# image_utils.py
from PIL import Image
from io import BytesIO
import os

# Thư mục tĩnh lưu trữ ảnh đã chụp
PHOTO_DIR = "E:\Project_AI\image_storage"


def save_captured_image(image_data, filename: str):
    """
    Lưu ảnh vào PHOTO_DIR với tên filename.
    image_data: có thể là bytes hoặc Streamlit UploadedFile.
    filename: tên file (ví dụ: RESU0001.jpg).
    """
    os.makedirs(PHOTO_DIR, exist_ok=True)
    # Lấy bytes
    raw = image_data.getvalue() if hasattr(image_data, 'getvalue') else image_data
    img = Image.open(BytesIO(raw))
    img.save(os.path.join(PHOTO_DIR, filename))
