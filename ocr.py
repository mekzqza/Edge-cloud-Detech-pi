import os
import re
import string
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz, process

# engine อ่านตัวอักษร — สลับได้ที่บรรทัด import บรรทัดเดียว ขอแค่มี recognize(BGR) → (text, score)
# ตอนนี้ใช้โมเดลทางการ th_PP-OCRv5_mobile_rec (rec_onnx.py ตัวที่ fine-tune เองแพ้ไปแล้ว ลบทิ้ง
# อยู่ใน git history ถ้าอยากได้กลับมา)
# ตัว TextRecognition(...) อยู่ใน rec_paddle._load() — สร้างตอนเรียกใช้ครั้งแรก ไม่ใช่ตอน import
# เพราะ Run_v2 --check / ocr.py --check ต้องรันได้โดยไม่มีโมเดล
import rec_paddle as rec

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
}

# ponytail: แยกเป็นคนละ map โดยตั้งใจ — THAI_MAP ใช้ทั้งโซนเลขทะเบียนและ clean_province_text
# จึงมีได้แค่ อังกฤษ/ตัวเลข → ไทย ส่วน PLATE_ONLY_MAP เป็น ไทย → ไทย ใช้เฉพาะเลขทะเบียน
# ถ้าปล่อยให้ "า"→"ว" ไปโดนชื่อจังหวัด: ตาก→ตวก คะแนน fuzzy ตกจาก 100 เหลือ 67 = หลุด
# PROVINCE_THRESHOLD (70) และทุกจังหวัดที่มีสระ า จะเหลือ margin บางลง 10-30 คะแนน
PLATE_ONLY_MAP = {"า": "ว", "ฤ": "ฎ"}

# ตัวอักษรอังกฤษ/สัญลักษณ์ที่ OCR มักอ่านแทนตัวเลข — ใช้เฉพาะโซนตัวเลขท้ายป้าย
DIGIT_MAP = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "i": "1", "m": "1",
    "Z": "2", "E": "3", "A": "4", "S": "5", "s": "5",
    "G": "6", "b": "6", "T": "7", "B": "8", "g": "9", "q": "9",
}

VALID_THAI_CONSONANTS = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")

# ตัวที่ยอมให้ผ่านเข้าไปซ่อมป้าย: เลข พยัญชนะไทย และทุกตัวที่มี map แปลงรออยู่
# ที่เหลือคือขยะจากภาพล้วนๆ — กรอบป้าย/น็อต ([ ] |) วรรณยุกต์ที่ป้ายไม่มี (๊ ่ ้ ๋)
# จุด ขีด เว้นวรรค — rec อ่านติดมาจากขอบ crop ไม่ใช่ตัวอักษรที่หายไป ทิ้งได้ปลอดภัย
#
# ponytail: ประกอบจาก map แทนที่จะเขียน regex ตายตัว — เพิ่มตัวสับสนใหม่ใน DIGIT_MAP/
# THAI_MAP แล้วมันผ่านฟิลเตอร์นี้เอง ไม่ต้องจำว่ามีอีกที่ให้แก้ (เขียน [^0-9ก-ฮA-Za-z]
# ตรงๆ จะกิน "า" ของ PLATE_ONLY_MAP ทิ้ง แล้ว า→ว ตายเงียบ)
#
# ต้องเก็บ "ทุก" ตัวอังกฤษและทุกตัวในช่วง ก-ฮ ไว้ แม้ตัวที่ไม่มี map (M, X, ฦ) — พวกนั้น
# อาจเป็นพยัญชนะที่อ่านเพี้ยน กรองทิ้งตรงนี้จะได้ป้ายสั้นลง 1 ตัวที่ "ตรงฟอร์แมต" แล้วกิน
# น้ำหนักเต็มตอนโหวต (1NX1234 → 1ก1234) = เปลี่ยนผิดแบบเห็นได้ เป็นผิดแบบมั่นใจ
# ปล่อยให้ _repair_letters เป็นคนตัดสินเหมือนเดิม (ดูคอมเมนต์ dropped ใน _try_fix_plate)
_PLATE_KEEP = (
    set(string.digits)
    | {chr(c) for c in range(0x0E01, 0x0E2F)}  # ก-ฮ ทั้งช่วง ไม่ใช่แค่ที่ใช้ได้จริง
    | set(string.ascii_letters)
    | set(DIGIT_MAP)
    | set(THAI_MAP)
    | set(PLATE_ONLY_MAP)
)

# ฟอร์แมตป้ายไทย: [เลขนำหน้า] + พยัญชนะ 1-3 ตัว + เลข 1-4 หลัก (ตรงกับที่ clean_plate_number ปั้นออกมา)
PLATE_PATTERN = re.compile(r"^[0-9]?[ก-ฮ]{1,3}[0-9]{1,4}$")
BAD_PATTERN_WEIGHT = 0.4  # ผิดฟอร์แมต = เหลือน้ำหนักเท่านี้ตอนโหวต

# แถวจังหวัดตัวเล็กกว่าแถวเลขเกือบครึ่ง score จาก rec เลยต่ำกว่าแถวบนเสมอ ใช้ threshold
# เดียวกัน (0.6) = ทิ้งจังหวัดทุกใบ — ปล่อยให้ fuzzy match กับรายชื่อ 77 จังหวัดเป็นตัวกรองแทน
# ค่านี้เอาไว้กันแค่ noise ล้วนๆ ปรับขึ้นถ้าเจอจังหวัดมั่วบ่อย
PROVINCE_CONF = 0.15

# วิธีแยกแถวบน (เลข) ออกจากแถวล่าง (จังหวัด)
#   "yolo"  = ให้ YOLO ชี้กล่อง number/province มาให้ — ค่าเริ่มต้น อยู่ระหว่างวัดผลจริง
#   "ratio" = หารที่ split_ratio ของความสูง — ของเดิม เร็วกว่า ไม่ต้องโหลดโมเดลเพิ่ม
#
# ย้าย default มา yolo วันที่ 2026-09-01 เพื่อเก็บผลรันจริงทั้งวัน — ยังไม่ใช่ข้อสรุป
# หลักฐานที่มีตอนนี้คือ ~/datatest 20 ใบ: 3 ใบที่หารครึ่งตัดพลาดกลับมาอ่านออก
# (007016→กว7016, nM940→กก940, ข๊8755→ขค8755) อีก 13 ใบที่ดีอยู่แล้วไม่พัง
# ocr score เฉลี่ยของใบที่อ่านออกทั้ง 2 โหมด 0.845 → 0.866
# ที่ยังไม่รู้: ยังไม่มีเฉลยว่าใบที่ "ดีขึ้น" ตรงป้ายจริงไหม และยังไม่ได้จับเวลาบน Pi
#
# ถอยกลับของเดิมได้ทันทีโดยไม่ต้องแก้โค้ด — ไม่ต้อง revert ไม่ต้อง restart service ทั้งชุด:
#   OCR_SPLIT=ratio python3 Run_v2.py
# ถ้าถอยถาวร ให้เปลี่ยนค่า default บรรทัดล่างนี้กลับเป็น "ratio"
SPLIT_MODE = os.environ.get("OCR_SPLIT", "yolo")

# IMGSZ ตรึงตามไฟล์ .onnx ([1,3,320,320]) เปลี่ยนขนาด = เปลี่ยนไฟล์โมเดล ต้องแก้คู่กันเสมอ
PARTS_MODEL = Path(__file__).parent / "model" / "License_Province_320" / "best.onnx"
PARTS_IMGSZ = 320
PARTS_CONF = 0.50

# ultralytics เห็น .onnx แล้วจะ pip install onnx ให้ ซึ่งบน Pi พังเสมอ (PEP 668)
# detect.py ตั้งไว้แล้วและถูก import ก่อนเสมอใน Run_v2 แต่ ocr.py ถูกเรียกเดี่ยวๆ ได้
# ต้องอยู่ก่อน import ultralytics ใน _parts_model() (ตั้งทีหลังไม่ทัน มันอ่านตอน import)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

_parts = None  # โหลดตอนใช้ครั้งแรก ไม่ใช่ตอน import — ocr.py --check ต้องรันได้โดยไม่มีโมเดล
_parts_broken = False  # โหลด/รันโมเดลพังแล้ว = เลิกลอง ถอยไป ratio ตลอดทั้งรอบ

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


def read_plate(
    image_source,
    split_ratio,  # ทำงาน 2
    province_threshold,
    conf_threshold,
):
    """อ่านป้ายใบเดียว → (เลขทะเบียน, จังหวัด, score ของแถวบน, score ของจังหวัด) — score ใช้เป็นน้ำหนักตอนโหวต"""

    if isinstance(image_source, (str, Path)):
        img = cv2.imread(str(image_source))
        if img is None:
            raise FileNotFoundError(f"ไม่พบไฟล์: {image_source}")
    elif isinstance(image_source, np.ndarray):
        img = image_source
    else:
        raise TypeError(
            f"image_source ต้องเป็น str/Path (path ไฟล์) หรือ np.ndarray (ภาพที่โหลดแล้ว) "
            f"แต่ได้รับ {type(image_source)}"
        )
    img = perspective_plate(img)

    # กล่องจาก YOLO ตัดพอดีตัวอักษรอยู่แล้ว จึงไม่ต้อง crop_border ต่อ — crop_border มีไว้
    # กันกรอบป้าย/น็อตเข้า rec engine ซึ่งเป็นปัญหาของการหารครึ่งตายตัวเท่านั้น
    top_img, bottom_img = split_by_yolo(img) if SPLIT_MODE == "yolo" else (None, None)
    if top_img is None:
        top_img, bottom_img = split_plate_image(crop_border(img, margin=0.04), split_ratio)

    # OCR พร้อม confidence threshold ตามงานวิจัยโรมาเนีย
    top_results = ocr_image(top_img, conf_threshold)
    plate_number_raw = " ".join(r[0] for r in top_results)
    license_id = clean_plate_number(plate_number_raw)
    bottom_results = ocr_image(bottom_img, PROVINCE_CONF)
    province_raw = " ".join(r[0] for r in bottom_results)
    province, prov_score = match_province(
        clean_province_text(province_raw), threshold=province_threshold
    )

    score = max((r[1] for r in top_results), default=0.0)
    # prov_score เป็น fuzzy ratio 0-100 → หาร 100 ให้อยู่สเกลเดียวกับ conf ตัวอื่น
    return license_id, province, score, prov_score / 100


def order_points(pts):  # ทำงาน 3
    """
    เรียงมุมเป็น
    TL, TR, BR, BL
    """

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]  # Top Left
    rect[2] = pts[np.argmax(s)]  # Bottom Right

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]  # Top Right
    rect[3] = pts[np.argmax(diff)]  # Bottom Left

    return rect


def perspective_plate(plate):  # หมุนภาพ

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # ลด Noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ถ้าป้ายเป็นตัวดำพื้นขาว
    # ลองเปิดบรรทัดนี้แทน
    # thresh = cv2.bitwise_not(thresh)

    # หา Contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contour")
        return plate

    # เลือก Contour ใหญ่สุด
    contour = max(contours, key=cv2.contourArea)

    # หา Min Area Rectangle
    rect = cv2.minAreaRect(contour)

    # 4 มุม
    box = cv2.boxPoints(rect)

    box = np.float32(box)

    # เรียงมุม
    box = order_points(box)

    # คำนวณความกว้าง
    widthA = np.linalg.norm(box[2] - box[3])

    widthB = np.linalg.norm(box[1] - box[0])

    maxWidth = int(max(widthA, widthB))

    # คำนวณความสูง
    heightA = np.linalg.norm(box[1] - box[2])

    heightB = np.linalg.norm(box[0] - box[3])

    maxHeight = int(max(heightA, heightB))

    # ponytail: contour ใหญ่สุด "ควร" เป็นตัวป้าย แต่พอภาพเบลอ/ย้อนแสง OTSU จะได้ก้อนมั่วๆ
    # แล้ว warp ทั้งป้ายลงไปในก้อนนั้น = แถวจังหวัดหายเกลี้ยง เล็กกว่า 50% ของ crop ถือว่าเชื่อไม่ได้
    h, w = plate.shape[:2]
    if maxWidth < 2 or maxHeight < 2 or maxWidth * maxHeight < 0.5 * h * w:
        return plate

    # จุดปลายทาง
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # Homography Matrix
    M = cv2.getPerspectiveTransform(box, dst)

    # Perspective Transform
    warp = cv2.warpPerspective(plate, M, (maxWidth, maxHeight))

    return warp


def crop_border(image, margin):  # ตัดขอบภาพทะเบียน ทำงาน 4

    h, w = image.shape[:2]

    # จำนวน pixel ที่ตัด
    mx = int(w * margin)
    my = int(h * margin)

    cropped = image[my : h - my, mx : w - mx]

    return cropped


def split_plate_image(image: np.ndarray, split_ratio: float):  # ทำงาน 5
    h, w = image.shape[:2]
    cut = int(h * split_ratio)
    return image[:cut, :], image[cut:, :]


def _parts_model():
    """โหลด YOLO แยกแถวครั้งเดียว — เรียกครั้งแรกช้า (โหลดไฟล์ + warm-up)

    warm-up ตรงนี้ยังเป็นตัวยืนยันว่า PARTS_IMGSZ ตรงกับไฟล์ .onnx ด้วย ไม่ตรงจะพัง
    ตอนอ่านป้ายใบแรก ไม่ใช่ตอนรถคันที่ร้อยผ่าน
    """
    global _parts
    if _parts is None:
        from ultralytics import YOLO

        _parts = YOLO(str(PARTS_MODEL), task="detect")
        _parts(
            np.zeros((PARTS_IMGSZ, PARTS_IMGSZ, 3), dtype=np.uint8),
            imgsz=PARTS_IMGSZ,
            verbose=False,
        )
    return _parts


def _crops_from_boxes(image, boxes):
    """boxes = {"number": (x1,y1,x2,y2), ...} → (crop แถวเลข, crop แถวจังหวัด)

    ไม่มีกล่อง number = เชื่อกล่องชุดนี้ไม่ได้ทั้งใบ คืน (None, None) ให้ฝั่งเรียก fallback
    ไปหารครึ่งตามเดิม — ไม่ผสม "เลขจาก YOLO + จังหวัดจาก ratio" เพราะพอผลเพี้ยนแล้ว
    ไล่ไม่ออกว่าใบนั้นมาทางไหน

    ขาดแค่ province ยังใช้ได้ (แถวล่างเลือน/โดนเฉือนเป็นเรื่องปกติ) — คืน None แล้ว
    ocr_image คืน [] เอง ไหลต่อเป็นจังหวัดว่างตามกลไกเดิม
    """
    h, w = image.shape[:2]

    def crop(name):
        if name not in boxes:
            return None
        x1, y1, x2, y2 = boxes[name]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None

    number = crop("number")
    return (number, crop("province")) if number is not None else (None, None)


def split_by_yolo(image):
    """แยกแถวด้วยกล่องจาก YOLO แทนการหารครึ่งตายตัว — เชื่อไม่ได้คืน (None, None)

    จับ Exception กว้างๆ โดยตั้งใจ: โหมดนี้มี ratio เป็นตาข่ายรองอยู่แล้ว ปล่อยให้ throw
    ขึ้นไปจะไปโดน except กว้างใน Run_v2.main() ซึ่ง "ย้ายก้อนออกจากคิว" = ไฟล์โมเดลหาย
    ครั้งเดียว รถตกทั้งวันเงียบๆ เหลือแค่บรรทัด ⚠️ อ่านป้ายไม่ได้ยังพอทน ทิ้งรถทั้งวันไม่ทน

    latch ไว้ด้วยเพราะโหลดโมเดลพังกินเวลาหลายวินาทีต่อครั้ง และ log จะท่วมทุกเฟรม
    """
    global _parts_broken

    if _parts_broken:
        return None, None

    try:
        model = _parts_model()
        results = model(image, conf=PARTS_CONF, imgsz=PARTS_IMGSZ, verbose=False)
    except Exception as e:
        _parts_broken = True
        print(f"[WARN] แยกแถวด้วย YOLO ใช้ไม่ได้ ({type(e).__name__}: {e}) — ใช้ ratio ต่อทั้งรอบ")
        return None, None

    best = {}

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            # ponytail: ป้ายใบเดียวมีแถวเลขแถวเดียว กล่องที่เกินมาคือ false positive
            # เอาเฉพาะตัวที่มั่นใจสุดต่อคลาส ไม่ใช่จ่าย OCR ให้ทุกกล่องที่โผล่มา
            name = model.names[int(box.cls[0])]
            score = float(box.conf[0])
            if score > best.get(name, (0.0, None))[0]:
                best[name] = (score, tuple(map(int, box.xyxy[0])))

    return _crops_from_boxes(image, {k: v[1] for k, v in best.items()})


def ocr_image(image: np.ndarray, conf_threshold: float = 0.6):  # ทำงาน 6
    """อ่าน OCR และกรองผลลัพธ์ที่ confidence ต่ำกว่า threshold ออก (ตามงานวิจัยโรมาเนีย)"""
    # None = YOLO ไม่เจอแถวจังหวัด, size 0 = crop_border/split ตัดจนไม่เหลือพิกเซล
    if image is None or image.size == 0:
        return []
    text, score = rec.recognize(image)
    return [(text, score)] if text and score >= conf_threshold else []


def clean_plate_number(text: str):
    """OCR ดิบ → เลขทะเบียนที่ตรงฟอร์แมต ซ่อมไม่ได้ก็คืนของดิบ

    ของดิบที่ผิดฟอร์แมตไม่ใช่ความล้มเหลว — plate_vote จะหักน้ำหนักเหลือ
    BAD_PATTERN_WEIGHT ให้เอง แล้วเฟรมอื่นที่อ่านออกจะชนะไปตามกลไก
    """
    # กรองขยะจากภาพทิ้งก่อนซ่อม — เว้นวรรคก็หลุดตรงนี้ด้วย ไม่ต้อง replace แยก
    raw = "".join(c for c in text if c in _PLATE_KEEP)
    if not raw:
        return ""
    return _try_fix_plate(raw) or raw


def _try_fix_plate(raw: str):
    """ซ่อมทีละโซน — คืน None ถ้าซ่อมแล้วยังไม่ตรงฟอร์แมต (อย่าเดาส่งๆ)"""
    number_zone, letter_zone = _split_number_letter_zone(raw)
    if number_zone is None:
        return None

    fixed_number = _repair_digits(number_zone)
    if not fixed_number.isdigit():
        return None

    # เลขนำหน้าป้าย (1กข1234) — "0" ไม่มีจริง ถือว่าเป็นตัวอักษรที่อ่านเพี้ยน
    prefix, letters = "", letter_zone
    if letters:
        head = DIGIT_MAP.get(letters[0], letters[0])
        if head.isdigit() and head != "0":
            prefix, letters = head, letters[1:]

    fixed_letters, dropped = _repair_letters(letters)

    # ponytail: มีตัวที่แปลไม่ออกโดนทิ้ง = เดาไม่ได้ว่าเดิมคือพยัญชนะตัวไหน ปล่อยผ่านจะได้
    # ป้ายสั้นลง 1 ตัวที่ "ตรงฟอร์แมต" (1กษ1234 → 1ก1234) แล้วกินน้ำหนักเต็มใน plate_vote
    # = เปลี่ยนผิดแบบเห็นได้ เป็นผิดแบบมั่นใจ ซึ่งพัง BAD_PATTERN_WEIGHT ทั้งกลไก
    if dropped:
        return None

    if len(fixed_letters) > 3:
        fixed_letters = _reduce_extra_letters(fixed_letters)

    plate = prefix + fixed_letters + fixed_number
    return plate if PLATE_PATTERN.match(plate) else None


def _split_number_letter_zone(raw: str):
    """หาจุดที่ตัวเลขท้ายป้ายเริ่ม → (โซนเลข, โซนตัวอักษร) — ไม่มีเลขท้ายเลยคืน (None, raw)"""
    idx = len(raw)
    for i in range(len(raw) - 1, -1, -1):
        if raw[i].isdigit() or raw[i] in DIGIT_MAP:
            idx = i
        else:
            break

    if idx == len(raw):
        return None, raw
    return raw[idx:], raw[:idx]


def _repair_digits(text: str):
    return "".join(DIGIT_MAP.get(c, c) for c in text)


def _repair_letters(text: str):
    """คืน (พยัญชนะไทยที่ซ่อมแล้ว, จำนวนตัวที่ทิ้งเพราะแปลไม่ออก)"""
    fixed = "".join(PLATE_ONLY_MAP.get(c, THAI_MAP.get(c, c)) for c in text)
    kept = "".join(c for c in fixed if c in VALID_THAI_CONSONANTS)
    return kept, len(fixed) - len(kept)


def _reduce_extra_letters(letters: str):
    """เกิน 3 ตัว = OCR เห็นตัวเดียวเป็นสองตัว — ยุบตัวซ้ำก่อน ไม่พอค่อยไล่ตัดทีละตัว

    ponytail: ไล่ตัดเป็น combinations = โตแบบ 2^n แต่ n คือความยาวแถวเลขป้าย
    (วัดจริง 12 ตัว = 4 ms) โตกว่านี้ไม่ได้อยู่แล้ว ถ้าวันหนึ่งช้าค่อยตัดที่ 8 ตัว
    """
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
            if 1 <= len(candidate) <= 3:
                return candidate

    return deduped[:3]


def clean_province_text(text: str) -> str:  # ทำงาน 8
    text = text.strip()
    for old, new in THAI_MAP.items():
        text = text.replace(old, new)
    text = re.sub(r"[^ก-๙]", "", text)
    return text


def match_province(text: str, threshold: int):  # ทำงาน 9
    if not text:
        return None, 0
    result = process.extractOne(
        text,
        THAI_PROVINCES,
        scorer=fuzz.token_set_ratio,
    )
    if result is None:
        return None, 0
    match, score, _ = result
    if score >= threshold:
        return match, score
    return None, score


def plate_vote(items):
    """โหวตทั้งสตริง — items = [(ข้อความที่ OCR ถอดได้, น้ำหนัก)] คืน (ป้ายที่ชนะ, คะแนนรวม)

    คะแนนต่อสตริง = ผลรวมน้ำหนักของทุกเฟรมที่อ่านออกมาได้เป็นสตริงนั้น สตริงที่ผิดฟอร์แมต
    ป้ายไทยหักเหลือ BAD_PATTERN_WEIGHT — เฟรมเดียวที่อ่านมั่วด้วย conf สูงจึงแพ้ป้ายฟอร์แมต
    ถูกที่โผล่ซ้ำ 2-3 เฟรม ซึ่งเป็นเหตุผลทั้งหมดของการโหวตข้ามเฟรม
    """
    tally = Counter()

    for text, weight in items:
        if text:
            bonus = 1.0 if PLATE_PATTERN.match(text) else BAD_PATTERN_WEIGHT
            tally[text] += weight * bonus

    if not tally:
        return "", 0.0

    best, score = tally.most_common(1)[0]

    # ทุกเฟรมผิดฟอร์แมตหมด = น่าจะพลาดคนละตัวอักษร ลองประกอบใหม่ทีละตัวจากเสียงส่วนใหญ่
    if not PLATE_PATTERN.match(best):
        merged = character_vote([t for t, _ in items])
        if PLATE_PATTERN.match(merged):
            return merged, score

    return best, score


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


def province_vote(provinces):

    provinces = [p for p in provinces if p]

    if not provinces:
        return None

    return Counter(provinces).most_common(1)[0][0]


def _selfcheck():
    """python3 ocr.py --check — เช็คตรรกะจังหวัด ไม่ต้องโหลดโมเดล"""
    junk = np.zeros((100, 200, 3), dtype=np.uint8)
    junk[40:50, 40:50] = 255  # ก้อนเล็กๆ ก้อนเดียว = contour เชื่อไม่ได้
    assert perspective_plate(junk).shape == junk.shape, "contour มั่วต้องคืนภาพเดิม ไม่ warp"

    # โหวตข้ามเฟรม: ป้ายที่โผล่ซ้ำชนะป้ายที่โผล่ครั้งเดียวแม้ conf จะสูงกว่า
    assert plate_vote([("1กข1234", 0.6), ("1กข1234", 0.6), ("1กข9999", 0.95)])[0] == "1กข1234"
    # ฟอร์แมตถูกชนะฟอร์แมตมั่วที่ conf สูงกว่า (คนละสตริงกัน โหวตกันไม่ได้ ต้องตัดสินด้วยฟอร์แมต)
    assert plate_vote([("1กข1234", 0.5), ("1ก1ข234!", 0.9)])[0] == "1กข1234"
    # ผิดฟอร์แมตทุกเฟรมแต่พลาดคนละตัว → ประกอบใหม่ทีละตัวอักษรต้องได้ป้ายที่ถูก
    assert plate_vote([("1กX1234", 0.8), ("1กข12X4", 0.5), ("Xกข1234", 0.5)])[0] == "1กข1234"
    assert plate_vote([])[0] == ""
    assert plate_vote([("", 0.9)])[0] == ""

    # ซ่อมเลขทะเบียน: ตัวอักษรที่อ่านแทนเลขต้องกลับมาเป็นเลข
    assert clean_plate_number("กท5O43") == "กท5043"
    assert clean_plate_number("1กข1234") == "1กข1234"  # ของถูกอยู่แล้วห้ามแตะ
    assert clean_plate_number("1ฤข1234") == "1ฎข1234"  # ฤ ไม่ใช่พยัญชนะป้าย → ฎ
    assert clean_plate_number("1กกขข1234") == "1กข1234"  # ขอบตัวอักษรทำให้เห็นเป็นตัวซ้ำ
    assert clean_plate_number("") == ""

    # ขยะจากภาพต้องถูกกรองทิ้งก่อนซ่อม ไม่ใช่ทำให้ทั้งใบผิดฟอร์แมตแล้วโดนหักน้ำหนัก
    assert clean_plate_number("ธ309]") == "ธ309"  # กรอบป้ายที่ rec อ่านติดมา
    assert clean_plate_number("ขย5535]") == "ขย5535"
    assert clean_plate_number("ข๊8755") == "ข8755"  # วรรณยุกต์ที่ป้ายจริงไม่มี
    assert clean_plate_number("1กข 1234") == "1กข1234"  # เว้นวรรคยังต้องหลุดเหมือนเดิม
    assert clean_plate_number("]|.-") == "", "ขยะล้วนต้องเหลือว่าง ไม่ใช่คืนขยะ"
    # ตัวที่มี map รออยู่ห้ามโดนกรองทิ้ง ไม่งั้นการซ่อมตายเงียบ
    assert clean_plate_number("1ฤข1234") == "1ฎข1234"  # ฤ อยู่ใน PLATE_ONLY_MAP
    assert clean_plate_number("กท5O43") == "กท5043"  # O อยู่ใน DIGIT_MAP
    # ตัวที่แปลไม่ออกห้ามถูกทิ้งเงียบๆ จนกลายเป็นป้ายสั้นที่ "ตรงฟอร์แมต" แล้วกินน้ำหนักเต็ม
    assert not PLATE_PATTERN.match(clean_plate_number("1NX1234")), "ผิดต้องดูออกว่าผิด"
    assert plate_vote([("1กข1234", 0.5), (clean_plate_number("1NX1234"), 0.9)])[0] == "1กข1234"

    # แยกแถวด้วยกล่อง YOLO — เช็คเฉพาะตรรกะ ไม่โหลดโมเดล
    plate_img = np.zeros((100, 200, 3), dtype=np.uint8)
    # ไม่มีแถวเลข = เชื่อกล่องชุดนี้ไม่ได้ ต้อง fallback ทั้งใบ ไม่ใช่ใช้จังหวัดตัวเดียว
    assert _crops_from_boxes(plate_img, {"province": (0, 60, 200, 100)}) == (None, None)
    top, bottom = _crops_from_boxes(plate_img, {"number": (10, 5, 190, 55)})
    assert top.shape[:2] == (50, 180) and bottom is None, "จังหวัดขาดได้ ยังต้องอ่านเลขต่อ"
    # กล่องหลุดขอบต้องถูก clamp ไม่ใช่ได้ crop ว่างแล้วเงียบ
    top, _ = _crops_from_boxes(plate_img, {"number": (-20, -10, 500, 300)})
    assert top.shape[:2] == (100, 200)
    assert ocr_image(None) == [], "แถวจังหวัดที่ YOLO ไม่เจอต้องไหลต่อได้ ไม่ใช่ throw"

    assert clean_province_text(" ชลบุรี ") == "ชลบุรี"
    # PLATE_ONLY_MAP ("า"→"ว") ต้องไม่รั่วมาโดนชื่อจังหวัด ไม่งั้น ตาก→ตวก แล้วหลุด threshold
    assert clean_province_text("ตาก") == "ตาก"
    assert match_province("ตาก", 70)[0] == "ตาก"
    assert match_province("กาญจนบุรี", 70)[0] == "กาญจนบุรี"
    assert clean_province_text("นนทบุรี!") == "นนทบุรี"  # อักขระที่ไม่ใช่ไทยต้องหลุดออก
    # ตัวท้ายหาย (ขอบล่างโดนเฉือน) ยังต้อง match ได้ — นี่คือเคสจริงที่เจอบ่อยสุด
    assert match_province("นนทบุร", 70)[0] == "นนทบุรี"
    assert match_province("xyz", 70)[0] is None

    # โมเดลหาย/โหลดไม่ขึ้น ต้องถอยไป ratio เงียบๆ ไม่ใช่โยน exception จนรถตกคิวทั้งวัน
    # (ทำท้ายสุดเพราะพัง PARTS_MODEL ทิ้งไว้)
    global PARTS_MODEL, _parts, _parts_broken
    PARTS_MODEL, _parts, _parts_broken = Path("no_such_model.onnx"), None, False
    assert split_by_yolo(plate_img) == (None, None), "โมเดลพังต้องถอยไป ratio"
    assert _parts_broken, "พังแล้วต้องจำไว้ ไม่ลองโหลดใหม่ทุกเฟรม"

    print("✅ selfcheck ผ่าน")


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _selfcheck()
