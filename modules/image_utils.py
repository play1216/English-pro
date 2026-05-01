import io

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None


def load_image_rgb(uploaded_file):
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    uploaded_file.seek(0)
    return np.array(image)


def load_image_bgr(uploaded_file):
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    if cv2 is not None:
        image_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        return cv2.imdecode(image_array, 1)

    rgb_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(rgb_image)[:, :, ::-1]
