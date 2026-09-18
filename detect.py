"""YOLO หากล่องรถ + ป้ายทะเบียนในภาพหนึ่งใบ → (cars, plates)

แต่ละตัวเป็น dict {"image": crop, "bbox": (x1,y1,x2,y2), "conf": score}

    python3 detect.py --check   # เช็คตรรกะ crop/enhance (ต้องมีโมเดล เพราะโหลดตอน import)
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ultralytics เห็นไฟล์ .onnx แล้วจะ pip install onnx ให้อัตโนมัติ ซึ่งบน Pi พังเสมอ (PEP 668)
# แล้ว retry 2 รอบทุกครั้งที่สตาร์ท — onnx ใช้ตอน export เท่านั้น inference ไม่ต้องใช้
#
# ชื่อตัวแปรคือ YOLO_AUTOINSTALL (ultralytics/utils/__init__.py: env_bool("YOLO_AUTOINSTALL", True))
# ไม่ใช่ ULTRALYTICS_AUTOINSTALL อย่างที่เคยเขียนไว้ — ตั้งผิดชื่อ = ไม่มีผลอะไรเลย เงียบสนิท
# ตั้งไว้ทั้ง 2 ชื่อเผื่อ ultralytics คนละรุ่นบน Pi กับเครื่อง dev เรียกคนละชื่อ
# ต้องอยู่ก่อนบรรทัด import ultralytics เท่านั้น ตั้งทีหลังไม่ทัน (มันอ่านค่าตอน import)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

from ultralytics import YOLO  # noqa: E402  ต้องอยู่หลังการตั้ง env ข้างบน

# path ผูกกับตัวไฟล์ ไม่ใช่ cwd — รันจากโฟลเดอร์ไหนก็หาโมเดลเจอ ไม่ต้อง chdir (ใช้กับ systemd ได้)
#
# IMGSZ ต้องตรงกับไฟล์ .onnx ไม่ใช่ปรับได้ตอนรัน — ONNX export ตรึง input shape ไว้ตายตัว
# (round_3/best.onnx = [1,3,640,640]) ส่งขนาดอื่นไป onnxruntime จะโยน INVALID_ARGUMENT ทันที
# เปลี่ยนขนาด = เปลี่ยนไฟล์โมเดล ต้องแก้ 2 บรรทัดนี้พร้อมกันเสมอ
MODEL = Path(__file__).parent / "model" / "round_3" / "best.onnx"
IMGSZ = 640

# เผื่อขอบรอบ bbox กี่ % ของด้านนั้น — 0 = ตัดตาม bbox เป๊ะ
#
# วัดแล้วบน ~/datatest 30 ใบ (2026-08-27): 0.08 ได้ 24/30, 0 ได้ 28/30
# ขอบเผื่อลากกรอบป้าย/น็อตเข้า rec engine ด้วย แล้วมันอ่านกรอบเป็น "[" "]" ติดมากับป้าย
# (ธ309] ขย5535] กบ5166]) ทั้งยังดัน ocr score ลงเกือบทุกใบ — 0 จึงเป็นค่าเริ่มต้น
# ยังปรับได้ตอนรันถ้าเจอกล้องที่ bbox ตัดตัวอักษรขาด:
#   DETECT_MARGIN=0.05 python3 Run_v2.py --batch ~/datatest
MARGIN_RATIO = float(os.environ.get("DETECT_MARGIN", 0.0))

# ปรับแสงก่อนเข้า YOLO — ปิดไว้เป็นค่าเริ่มต้นโดยตั้งใจ ดูเหตุผลที่ enhance_image()
# เปิดทดสอบ: DETECT_ENHANCE=1 python3 Run_v2.py --batch โฟลเดอร์ IN
ENHANCE = os.environ.get("DETECT_ENHANCE") == "1"

TARGET_BRIGHTNESS = 130  # ความสว่างเฉลี่ยที่อยากได้ (0-255)
GAMMA_RANGE = (0.4, 2.5)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

model = YOLO(str(MODEL), task="detect")

# Warm-up: เรียกแรกช้ากว่าปกติหลายเท่า กินไปตั้งแต่ตอนโหลด ไม่ใช่ตอนรถคันแรกผ่าน
# และเป็นตัวยืนยันว่า IMGSZ ตรงกับไฟล์ — ไม่ตรงจะพังตรงนี้ตอน import ไม่ใช่ตอนรถคันแรกผ่าน
model(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), imgsz=IMGSZ, verbose=False)


def adaptive_gamma(image):
    """ดันความสว่างเฉลี่ยเข้าหา TARGET_BRIGHTNESS — ภาพมืดได้ gamma < 1 ภาพโอเวอร์ได้ > 1"""
    mean = max(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0, 0.01)
    gamma = np.clip(np.log(TARGET_BRIGHTNESS / 255.0) / np.log(mean), *GAMMA_RANGE)

    table = (((np.arange(256) / 255.0) ** gamma) * 255).astype(np.uint8)
    return cv2.LUT(image, table)


def enhance_image(image):
    """Adaptive Gamma + CLAHE บน L-channel

    ponytail: ปิดไว้เป็นค่าเริ่มต้นเพราะ best.onnx เทรนด้วยภาพดิบ — ป้อนภาพที่ปรับแล้ว
    คือเปลี่ยน distribution ที่โมเดลไม่เคยเห็น (domain shift) detection อาจตกแทนที่จะขึ้น
    และ CLAHE บน LAB ทั้งเฟรมกิน CPU ทุกเฟรม ซึ่งงบไม่เหลือ (ฝั่งอ่าน ~0.56 วิ/รูป ฝั่งถ่าย
    ป้อน 0.5 วิ/รูป) เปิดด้วย DETECT_ENHANCE=1 แล้ววัดด้วย --batch ว่าอ่านถูกขึ้นจริงไหม
    ก่อนตัดสินใจย้ายมาเป็นค่าเริ่มต้น
    """
    lab = cv2.cvtColor(adaptive_gamma(image), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def crop_with_margin(image, x1, y1, x2, y2, margin_ratio=MARGIN_RATIO):
    """crop พร้อมเผื่อขอบ — ตัดจนไม่เหลือพิกเซลคืน None

    ขอบสำคัญกับป้าย 2 ทาง: ตัวอักษรริมป้ายไม่โดนเฉือน และ perspective_plate() ใน ocr.py
    ต้องเห็นขอบป้ายถึงจะหา contour เจอ (crop ชิด bbox = contour ใหญ่สุดคือกรอบภาพเอง)
    """
    h, w = image.shape[:2]
    mx = int((x2 - x1) * margin_ratio)
    my = int((y2 - y1) * margin_ratio)

    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)

    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def detect(image, conf=0.70):
    """ponytail: inference ธรรมดา ไม่ track — ByteTrack ต้องใช้ lap ซึ่งลงบน Pi OS ไม่ได้
    (PEP 668) และฝั่งเรียกใช้จัดกลุ่มเฟรมของรถคันเดียวกันด้วยเวลาถ่าย (group_frames) แทน
    track id อยู่แล้ว กลับมาใส่ track ตอนย้ายไปอ่านสตรีมต่อเนื่องแล้วลง lap ได้
    """
    if ENHANCE:
        image = enhance_image(image)

    results = model(image, conf=conf, imgsz=IMGSZ, verbose=False)

    cars = []
    plates = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_name = model.names[int(box.cls[0])]
            if class_name not in ("car", "license-plate"):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = crop_with_margin(image, x1, y1, x2, y2)
            if crop is None:
                continue

            det = {"image": crop, "bbox": (x1, y1, x2, y2), "conf": float(box.conf[0])}
            (cars if class_name == "car" else plates).append(det)

    return cars, plates


def _selfcheck():
    """python3 detect.py --check"""
    img = np.full((100, 200, 3), 60, dtype=np.uint8)

    # เผื่อขอบแล้วต้องใหญ่ขึ้นจริง และไม่ทะลุขอบภาพ
    c = crop_with_margin(img, 50, 40, 150, 80, 0.08)  # ตรึงค่า ไม่งั้น DETECT_MARGIN ทำเทสพัง
    assert c.shape[:2] == (46, 116), c.shape  # สูง 40+2*3, กว้าง 100+2*8
    assert crop_with_margin(img, 0, 0, 200, 100).shape == img.shape, "ติดขอบต้องไม่ล้น"
    assert crop_with_margin(img, 10, 10, 10, 10) is None, "กว้าง 0 ต้องคืน None"

    # ภาพมืดต้องสว่างขึ้น ภาพสว่างจ้าต้องมืดลง และขนาด/ชนิดต้องไม่เปลี่ยน
    dark = np.full((60, 60, 3), 30, dtype=np.uint8)
    out = enhance_image(dark)
    assert out.shape == dark.shape and out.dtype == np.uint8
    assert np.mean(out) > np.mean(dark), "ภาพมืดต้องถูกดันให้สว่างขึ้น"
    bright = np.full((60, 60, 3), 230, dtype=np.uint8)
    assert np.mean(enhance_image(bright)) < np.mean(bright), "ภาพโอเวอร์ต้องถูกลดลง"

    print("✅ selfcheck ผ่าน")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _selfcheck()
