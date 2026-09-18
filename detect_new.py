from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL = (
    Path("model") / "round_3" / "best.onnx"
)  # Path Models ถ้าจะเปลี่ยนโมเดล 640 ให้ใช้ model/Final_Detect_640

model = YOLO(str(MODEL), task="detect")


# ============================================================
# CONFIG
# ============================================================

YOLO_IMG_SIZE = 320  # เปลี่ยนเป็น 640

DEFAULT_CONF = 0.70

TARGET_PLATE_CLASS = "license-plate"

MAX_ROTATION_ANGLE = 25.0


# ============================================================
# WARM-UP
# ============================================================

_dummy = np.zeros((320, 320, 3), dtype=np.uint8)  # เปลี่ยนเป็น 640 , 640

model(_dummy, imgsz=YOLO_IMG_SIZE, verbose=False)


# ============================================================
# IMAGE ENHANCEMENT CONFIG
# ============================================================

TARGET_BRIGHTNESS = 130

MIN_GAMMA = 0.4
MAX_GAMMA = 2.5

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)


# ============================================================
# ADAPTIVE GAMMA
# ============================================================


def adaptive_gamma(image):
    """
    ปรับ Gamma ตามความสว่างเฉลี่ยของภาพ
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)

    mean_normalized = max(mean_brightness / 255.0, 0.01)

    target_normalized = TARGET_BRIGHTNESS / 255.0

    gamma = np.log(target_normalized) / np.log(mean_normalized)

    gamma = np.clip(gamma, MIN_GAMMA, MAX_GAMMA)

    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(
        np.uint8
    )

    return cv2.LUT(image, table)


# ============================================================
# CLAHE
# ============================================================


def apply_clahe(image):
    """
    เพิ่ม Local Contrast ด้วย CLAHE
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# ENHANCE IMAGE
# ============================================================


def enhance_image(image):
    """
    Adaptive Gamma + CLAHE
    """

    if image is None:
        raise ValueError("image is None")

    gamma_image = adaptive_gamma(image)

    enhanced_image = apply_clahe(gamma_image)

    return enhanced_image


# ============================================================
# CROP WITH MARGIN
# ============================================================


def crop_with_margin(image, x1, y1, x2, y2, margin_ratio=0.08):
    """
    Crop object พร้อมเพิ่มขอบเล็กน้อย
    """

    height, width = image.shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = int(box_width * margin_ratio)

    margin_y = int(box_height * margin_ratio)

    x1 = max(0, x1 - margin_x)

    y1 = max(0, y1 - margin_y)

    x2 = min(width, x2 + margin_x)

    y2 = min(height, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2].copy()


# ============================================================
# ESTIMATE ROTATION ANGLE
# ============================================================


def estimate_rotation_angle(image):
    """
    ประมาณมุมเอียงของ license plate
    โดยใช้ Edge + Contour + minAreaRect
    """

    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]

    best_rect = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.10:
            continue

        rect = cv2.minAreaRect(contour)

        _, (width, height), _ = rect

        if width <= 1 or height <= 1:
            continue

        aspect_ratio = max(width, height) / min(width, height)

        # License plate ต้องเป็นแนวนอนโดยทั่วไป
        if aspect_ratio < 1.5:
            continue

        if area > best_score:
            best_score = area
            best_rect = rect

    if best_rect is None:
        return 0.0

    _, (width, height), angle = best_rect

    if width < height:
        angle += 90

    while angle > 90:
        angle -= 180

    while angle < -90:
        angle += 180

    # ป้องกัน contour ผิด
    if abs(angle) > MAX_ROTATION_ANGLE:
        return 0.0

    return float(angle)


# ============================================================
# ROTATE IMAGE
# ============================================================


def rotate_image(image, angle):
    """
    หมุนภาพโดยพยายามไม่ตัดขอบ
    """

    if image is None or image.size == 0:
        return image

    if abs(angle) < 0.5:
        return image.copy()

    height, width = image.shape[:2]

    center = (width / 2, height / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])

    sin = abs(matrix[0, 1])

    new_width = int(height * sin + width * cos)

    new_height = int(height * cos + width * sin)

    matrix[0, 2] += new_width / 2 - center[0]

    matrix[1, 2] += new_height / 2 - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ============================================================
# ROTATION CORRECTION
# ============================================================


def correct_plate_rotation(plate):
    """
    แก้ภาพ license plate ให้ตรง
    """

    angle = estimate_rotation_angle(plate)

    corrected = rotate_image(plate, angle)

    return corrected, angle


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================


def detect(image, conf=DEFAULT_CONF):
    """
    Pipeline:

    Input
        ↓
    Adaptive Gamma + CLAHE
        ↓
    YOLO Detection (320)
        ↓
    car + license-plate
        ↓
    Crop
        ↓
    Rotation Correction
        ↓
    return cars, plates


    Returns
    -------
    cars : list
        รายการรถที่ตรวจพบ

    plates : list
        รายการป้ายทะเบียนที่ตรวจพบ
    """

    if image is None:
        raise ValueError("image is None")

    # ========================================================
    # 1. IMAGE ENHANCEMENT
    # ========================================================

    enhanced = enhance_image(image)

    # ========================================================
    # 2. YOLO DETECTION
    # ========================================================

    results = model.predict(enhanced, conf=conf, imgsz=YOLO_IMG_SIZE, verbose=False)

    cars = []
    plates = []

    # ========================================================
    # 3. PROCESS DETECTIONS
    # ========================================================

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])

            class_name = model.names[cls]

            score = float(box.conf[0])

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ป้องกัน bbox ออกนอกภาพ
            x1 = max(0, x1)

            y1 = max(0, y1)

            x2 = min(enhanced.shape[1], x2)

            y2 = min(enhanced.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # =================================================
            # CAR
            # =================================================

            if class_name == "car":
                crop = crop_with_margin(enhanced, x1, y1, x2, y2)

                if crop is None:
                    continue

                cars.append({"image": crop, "bbox": (x1, y1, x2, y2), "conf": score})

            # =================================================
            # LICENSE PLATE
            # =================================================

            elif class_name == TARGET_PLATE_CLASS:
                crop = crop_with_margin(enhanced, x1, y1, x2, y2)

                if crop is None:
                    continue

                # ---------------------------------------------
                # Rotation Correction
                # ---------------------------------------------

                corrected, angle = correct_plate_rotation(crop)

                plates.append(
                    {
                        "image": corrected,
                        "bbox": (x1, y1, x2, y2),
                        "conf": score,
                        "rotation_angle": angle,
                    }
                )

    return cars, plates
