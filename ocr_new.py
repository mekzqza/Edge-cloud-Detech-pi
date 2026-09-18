import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from paddleocr import TextRecognition
from rapidfuzz import fuzz, process
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

YOLO2_MODEL_PATH = Path("model") / "yolo2" / "best.onnx"  # เปลี่ยนโมเดลเป็น 640
YOLO2_IMG_SIZE = 320  # เปลี่ยนเป็น 640
YOLO2_CONF = 0.50
OCR_MODEL_NAME = "th_PP-OCRv5_mobile_rec"


# ============================================================
# LOAD MODEL
# ============================================================

yolo2 = YOLO(str(YOLO2_MODEL_PATH), task="detect")
ocr = TextRecognition(model_name=OCR_MODEL_NAME)


# ============================================================
# WARM-UP
# ============================================================

_dummy_yolo = np.zeros((YOLO2_IMG_SIZE, YOLO2_IMG_SIZE, 3), dtype=np.uint8)

yolo2.predict(_dummy_yolo, imgsz=YOLO2_IMG_SIZE, conf=YOLO2_CONF, verbose=False)


_dummy_ocr = np.zeros((48, 320, 3), dtype=np.uint8)

for _ in ocr.predict(_dummy_ocr):
    pass


# ============================================================
# THAI CONFUSION MAP
# ============================================================

THAI_MAP = {
    "@": "ฮ",
    "&": "ฃ",
    "N": "ก",
    "n": "ก",
    "1": "ก",
    "0": "ค",
    "H": "ฬ",
    "W": "พ",
    "U": "ข",
    "A": "ฎ",
    "า": "ว",
    "ฤ": "ฎ",
}


# ============================================================
# DIGIT CONFUSION MAP
# ============================================================

DIGIT_MAP = {
    "O": "0",
    "o": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "l": "1",
    "i": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "s": "5",
    "G": "6",
    "b": "6",
    "T": "7",
    "B": "8",
    "g": "9",
    "q": "9",
    "m": "1",
}


# ============================================================
# VALID THAI CONSONANTS
# ============================================================

VALID_THAI_CONSONANTS = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")


# ============================================================
# PLATE PATTERN
# ============================================================

PLATE_PATTERN = re.compile(r"^[0-9]?[ก-ฮ]{1,3}\s?[0-9]{1,4}$")


# ============================================================
# THAI PROVINCES
# ============================================================

THAI_PROVINCES = [
    "กรุงเทพมหานคร",
    "กระบี่",
    "กาญจนบุรี",
    "กาฬสินธุ์",
    "กำแพงเพชร",
    "ขอนแก่น",
    "จันทบุรี",
    "ฉะเชิงเทรา",
    "ชลบุรี",
    "ชัยนาท",
    "ชัยภูมิ",
    "ชุมพร",
    "เชียงราย",
    "เชียงใหม่",
    "ตรัง",
    "ตราด",
    "ตาก",
    "นครนายก",
    "นครปฐม",
    "นครพนม",
    "นครราชสีมา",
    "นครศรีธรรมราช",
    "นครสวรรค์",
    "นนทบุรี",
    "นราธิวาส",
    "น่าน",
    "บึงกาฬ",
    "บุรีรัมย์",
    "ปทุมธานี",
    "ประจวบคีรีขันธ์",
    "ปราจีนบุรี",
    "ปัตตานี",
    "พระนครศรีอยุธยา",
    "พะเยา",
    "พังงา",
    "พัทลุง",
    "พิจิตร",
    "พิษณุโลก",
    "เพชรบุรี",
    "เพชรบูรณ์",
    "แพร่",
    "ภูเก็ต",
    "มหาสารคาม",
    "มุกดาหาร",
    "แม่ฮ่องสอน",
    "ยโสธร",
    "ยะลา",
    "ร้อยเอ็ด",
    "ระนอง",
    "ระยอง",
    "ราชบุรี",
    "ลพบุรี",
    "ลำปาง",
    "ลำพูน",
    "เลย",
    "ศรีสะเกษ",
    "สกลนคร",
    "สงขลา",
    "สตูล",
    "สมุทรปราการ",
    "สมุทรสงคราม",
    "สมุทรสาคร",
    "สระแก้ว",
    "สระบุรี",
    "สิงห์บุรี",
    "สุโขทัย",
    "สุพรรณบุรี",
    "สุราษฎร์ธานี",
    "สุรินทร์",
    "หนองคาย",
    "หนองบัวลำภู",
    "อ่างทอง",
    "อำนาจเจริญ",
    "อุดรธานี",
    "อุตรดิตถ์",
    "อุทัยธานี",
    "อุบลราชธานี",
    "เบตง",
]


# ============================================================
# YOLO #2 DETECTION
#
# Class:
# 0 = number
# 1 = province
# ============================================================


def detect_plate_parts(plate_image, conf=YOLO2_CONF):

    if plate_image is None:
        raise ValueError("plate_image is None")

    if plate_image.size == 0:
        return [], []

    results = yolo2.predict(plate_image, conf=conf, imgsz=YOLO2_IMG_SIZE, verbose=False)

    plate_numbers = []
    provinces = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])

            # ====================================================
            # YOLO #2 CLASS
            #
            # 0 = number
            # 1 = province
            # ====================================================

            class_name = yolo2.names[cls]

            score = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ----------------------------------------------------
            # ป้องกัน Bounding Box หลุดภาพ
            # ----------------------------------------------------

            x1 = max(0, x1)

            y1 = max(0, y1)

            x2 = min(plate_image.shape[1], x2)

            y2 = min(plate_image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = plate_image[y1:y2, x1:x2].copy()

            detection = {"image": crop, "bbox": (x1, y1, x2, y2), "conf": score}

            # ====================================================
            # CLASS 0
            # number
            # ====================================================

            if class_name == "number":
                plate_numbers.append(detection)

            # ====================================================
            # CLASS 1
            # province
            # ====================================================

            elif class_name == "province":
                provinces.append(detection)

    return (plate_numbers, provinces)


# ============================================================
# READ ONE PLATE
# ============================================================


def read_plate(image_source, province_threshold, conf_threshold, yolo2_conf=YOLO2_CONF):

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    if isinstance(image_source, (str, Path)):
        img = cv2.imread(str(image_source))

        if img is None:
            raise FileNotFoundError(f"ไม่พบไฟล์: {image_source}")

    elif isinstance(image_source, np.ndarray):
        img = image_source

    else:
        raise TypeError("image_source ต้องเป็น str/Path หรือ np.ndarray")

    # ========================================================
    # YOLO #2
    # ========================================================

    (plate_number_images, province_images) = detect_plate_parts(img, conf=yolo2_conf)

    # ========================================================
    # OCR NUMBER
    # ========================================================

    license_results = []

    for detection in plate_number_images:
        crop = detection["image"]

        results = ocr_image(crop, conf_threshold)

        for text, score in results:
            license_id = clean_plate_number(text)

            if license_id:
                license_results.append(license_id)

    # ========================================================
    # OCR PROVINCE
    # ========================================================

    province_results = []

    for detection in province_images:
        crop = detection["image"]

        results = ocr_image(crop, conf_threshold)

        for text, score in results:
            province_raw = clean_province_text(text)

            province, prov_score = match_province(
                province_raw, threshold=province_threshold
            )

            if province:
                province_results.append(province)

    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_license = character_vote(license_results)

    final_province = province_vote(province_results)

    return (final_license, final_province)


# ============================================================
# READ MULTIPLE FRAMES
# ============================================================


def read_plate_sequence(
    images, province_threshold, conf_threshold, yolo2_conf=YOLO2_CONF
):

    license_results = []
    province_results = []

    for img in images:
        license_id, province = read_plate(
            img, province_threshold, conf_threshold, yolo2_conf
        )

        if license_id:
            license_results.append(license_id)

        if province:
            province_results.append(province)

    final_license = character_vote(license_results)

    final_province = province_vote(province_results)

    return (final_license, final_province)


# ============================================================
# OCR RESULT PARSER
# ============================================================


def parse_ocr_result(res):

    if isinstance(res, dict):
        data = res.get("res", res)

        return (data.get("rec_text", ""), data.get("rec_score", 0.0))

    if hasattr(res, "rec_text"):
        return (res.rec_text, getattr(res, "rec_score", 0.0))

    if hasattr(res, "__dict__"):
        d = vars(res)

        return (d.get("rec_text", ""), d.get("rec_score", 0.0))

    return (str(res), 0.0)


# ============================================================
# OCR IMAGE
# ============================================================


def ocr_image(image: np.ndarray, conf_threshold: float = 0.6):

    if image is None:
        return []

    if image.size == 0:
        return []

    output = []

    for res in ocr.predict(image):
        text, score = parse_ocr_result(res)

        if score >= conf_threshold:
            output.append((text, score))

    return output


# ============================================================
# CLEAN PLATE NUMBER
# ============================================================


def clean_plate_number(text: str):

    raw = text.replace(" ", "").strip()

    if not raw:
        return ""

    fixed = _try_fix_plate(raw)

    if fixed:
        return fixed

    print(f"[WARN] ไม่สามารถยืนยันรูปแบบป้ายทะเบียนได้: '{raw}'")

    return raw


# ============================================================
# TRY FIX PLATE
# ============================================================


def _try_fix_plate(raw: str):

    number_zone, letter_zone = _split_number_letter_zone(raw)

    if number_zone is None:
        return None

    fixed_number = _repair_digits(number_zone)

    if not fixed_number or not fixed_number.isdigit():
        return None

    prefix = ""

    letters = letter_zone

    if letters:
        first = letters[0]

        if first.isdigit() and first != "0":
            prefix = first

            letters = letters[1:]

        elif first in DIGIT_MAP and DIGIT_MAP[first] != "0":
            prefix = DIGIT_MAP[first]

            letters = letters[1:]

    fixed_letters = _repair_letters(letters)

    if len(fixed_letters) > 3:
        fixed_letters = _reduce_extra_letters(fixed_letters)

    plate = prefix + fixed_letters + fixed_number

    if PLATE_PATTERN.match(plate):
        return plate

    return None


# ============================================================
# SPLIT NUMBER / LETTER
# ============================================================


def _split_number_letter_zone(raw: str):

    idx = len(raw)

    for i in range(len(raw) - 1, -1, -1):
        ch = raw[i]

        if ch.isdigit() or ch in DIGIT_MAP:
            idx = i

        else:
            break

    if idx == len(raw):
        return (None, raw)

    return (raw[idx:], raw[:idx])


# ============================================================
# DIGIT REPAIR
# ============================================================


def _repair_digits(text: str):

    return "".join(DIGIT_MAP.get(c, c) for c in text)


# ============================================================
# THAI LETTER REPAIR
# ============================================================


def _repair_letters(text: str):

    fixed = "".join(THAI_MAP.get(c, c) for c in text)

    return "".join(c for c in fixed if c in VALID_THAI_CONSONANTS)


# ============================================================
# REDUCE EXTRA LETTERS
# ============================================================


def _reduce_extra_letters(letters: str):

    if not letters:
        return ""

    deduped = letters[0]

    for c in letters[1:]:
        if c != deduped[-1]:
            deduped += c

    if len(deduped) <= 3:
        return deduped

    for drop_count in range(1, len(deduped) - 2):
        for positions in combinations(range(len(deduped)), drop_count):
            candidate = "".join(c for i, c in enumerate(deduped) if i not in positions)

            if 1 <= len(candidate) <= 3 and all(
                c in VALID_THAI_CONSONANTS for c in candidate
            ):
                return candidate

    return deduped[:3]


# ============================================================
# CLEAN PROVINCE
# ============================================================


def clean_province_text(text: str):

    text = text.strip()

    for old, new in THAI_MAP.items():
        text = text.replace(old, new)

    text = re.sub(r"[^ก-๙]", "", text)

    return text


# ============================================================
# PROVINCE MATCHING
# ============================================================


def match_province(text: str, threshold: int):

    if not text:
        return None, 0

    result = process.extractOne(text, THAI_PROVINCES, scorer=fuzz.token_set_ratio)

    if result is None:
        return None, 0

    match, score, _ = result

    if score >= threshold:
        return (match, score)

    return (None, score)


# ============================================================
# CHARACTER VOTING
# ============================================================


def character_vote(texts):

    texts = [t for t in texts if t]

    if not texts:
        return ""

    max_len = max(len(t) for t in texts)

    result = ""

    for i in range(max_len):
        chars = []

        for t in texts:
            if i < len(t):
                chars.append(t[i])

        if chars:
            result += Counter(chars).most_common(1)[0][0]

    return result


# ============================================================
# PROVINCE VOTING
# ============================================================


def province_vote(provinces):

    provinces = [p for p in provinces if p]

    if not provinces:
        return None

    return Counter(provinces).most_common(1)[0][0]
