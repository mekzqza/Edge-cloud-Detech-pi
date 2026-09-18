"""ฝั่งอ่านป้าย — เฝ้า unsend/ จับเฟรมของรถคันเดียวกันมาโหวต แล้วยิงขึ้น backend ครั้งเดียว

รันคู่กับ Run_v2_capture.py (คนละโปรเซส):

    python3 Run_v2_capture.py   # ฝั่งถ่าย
    python3 Run_v2.py           # ฝั่งอ่าน + ส่ง
    python3 Run_v2.py --check   # เช็คตรรกะ ไม่ต้องมี YOLO/Paddle/กล้อง/เน็ต
    python3 Run_v2.py --batch โฟลเดอร์      # รันทั้งโฟลเดอร์ ลง Yolo_log.txt ไม่ยิง backend
    python3 Run_v2.py --batch โฟลเดอร์ IN   # เอารูปเก่ามาทดสอบเฉพาะกล้องขาเข้า
    python3 Run_v2.py --batch โฟลเดอร์ --score truth.csv   # เทียบกับเฉลย ตอบว่าอ่านถูกกี่ %
    python3 Run_v2.py --lag                # ฝั่งอ่านตามหลังฝั่งถ่ายกี่วิ (ไม่โหลดโมเดล)
"""

import base64
import csv
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# หมายเหตุ: การปิด autoinstall ของ ultralytics ย้ายไปอยู่ใน detect.py แล้ว (ต้องตั้งก่อน
# import ultralytics ซึ่งเกิดที่นั่นที่เดียว) ที่นี่ตั้งไม่ทันเพราะ detect ถูก import แบบ lazy

# ===== ตั้งค่า =====
API_URL = "https://edge-cloud-detect.sukpat.dev/api/detections"
UNSEND_DIR = Path("/home/pi/pi-edge_cloud/unsend")  # โฟลเดอร์รูปรอส่ง
SENT_DIR = Path("/home/pi/pi-edge_cloud/send")  # ย้ายมาที่นี่เมื่อส่งสำเร็จ
POLL_INTERVAL = 3  # วนเช็คทุกกี่วินาที
MAX_SENT_FILES = 200  # เก็บรูปที่ส่งแล้วใน send/ ไว้กี่รูป เกินแล้วลบตัวเก่าสุดทิ้ง
MAX_UNSEND_FILES = 200  # คิวรอส่งค้างได้กี่รูป เกินแล้วทิ้งตัวเก่าสุด (กันดิสก์เต็มตอนรันทั้งคืน)
IMG_EXTS = {".jpg", ".jpeg", ".png"}
YOLO_LOG = Path("/home/pi/Yolo_log.txt")  # log เฉพาะตอน YOLO เจอป้าย
FAILED_LOG = Path("/home/pi/filed_to_send.txt")  # log ตอนส่งขึ้น backend ไม่สำเร็จ

DETECT_CONF = 0.3  # default ของ detect.py คือ 0.70 สูงไปสำหรับป้ายไกลๆ
# ป้ายที่กว้างไม่ถึง ~40 px หลังย่อเป็น imgsz แล้ว YOLO มองไม่เห็น — กล้องที่ติดไกลกว่า
# ต้องใช้ imgsz ใหญ่ขึ้นแลกกับเวลา แต่ปรับตอนรันไม่ได้: ไฟล์ .onnx ตรึง input shape ไว้ตายตัว
# เปลี่ยนขนาด = เปลี่ยนไฟล์โมเดล แก้ MODEL + IMGSZ ใน detect.py พร้อมกัน
SPLIT_RATIO = 0.60  # แบ่งป้าย บน=เลขทะเบียน ล่าง=จังหวัด
PROVINCE_THRESHOLD = 70  # คะแนน fuzzy match จังหวัดขั้นต่ำ
OCR_CONF = 0.6  # ทิ้งผล OCR ที่ score ต่ำกว่านี้

VOTE_N = 7  # โหวตจากกี่เฟรมต่อรถหนึ่งคัน
# หยิบเฟรมจากฝั่งไหนของก้อนมาอ่าน — เป็นสมบัติของ "วิธีติดกล้อง" ไม่ใช่ของรูป
#   "head" = เฟรมแรกๆ  สำหรับกล้องที่รถวิ่งลอดใต้กล้องไป ป้ายพ้นขอบล่างเร็ว (IN)
#   "tail" = เฟรมท้ายๆ  สำหรับกล้องที่รถวิ่งเข้าหาแล้วจอด/ผ่านช้าๆ ป้ายใหญ่ขึ้นเรื่อยๆ (OUT)
# ตั้งผิดฝั่ง = จ่าย OCR ครบทุกเฟรมแต่ได้แต่ภาพฝากระโปรง รถหายทั้งคันแบบไม่มี log
# จูนผ่าน env ได้: VOTE_FROM_IN=tail python3 Run_v2.py --batch send IN
VOTE_FROM = {"IN": "head", "OUT": "head"}
VOTE_FROM_DEFAULT = "tail"  # กล้องที่ไม่ได้ตั้งไว้/ไฟล์เก่าที่เป็น UNKNOWN
GROUP_GAP = 3.0  # เฟรมห่างกันเกินกี่วินาที = คนละคัน (ฝั่งถ่ายเซฟทุก 0.5 วิตอนเห็นการเคลื่อนไหว)
POST_COOLDOWN = 60  # ป้ายเดิม+กล้องเดิม ยิงซ้ำภายในกี่วินาที = ถือว่ารถคันเดิม ข้ามไป
MAX_GROUP = 12  # ก้อนโตเกินนี้บังคับโหวตเลย — กันค้างตอนภาพไหวไม่หยุด (ฝนตก/ใบไม้/ไฟกะพริบ)
# ==================

_pipeline = None
_ocr_cache = {}
_failed_logged = set()  # ไฟล์ที่ log ว่าส่งไม่สำเร็จไปแล้ว — กัน log ซ้ำทุกรอบ retry
_last_post = {}  # (กล้อง, ป้าย) → เวลาที่ถ่ายรูปที่ยิงสำเร็จล่าสุด — กันยิงซ้ำรถคันเดิม


def pipeline():
    """โหลด YOLO + Paddle ครั้งแรกที่ใช้ — แยกมาไว้ตรงนี้ให้ --check รันได้โดยไม่ต้องมีของหนัก"""
    global _pipeline
    if _pipeline is None:
        from detect import detect
        from ocr import read_plate

        _pipeline = (detect, read_plate)
    return _pipeline


def camera_of(img_path: Path) -> str:
    """pk ของกล้องจาก prefix ชื่อไฟล์ที่ฝั่งถ่ายติดมา (IN_20260812_...jpg → IN)

    ponytail: อ่านจากชื่อไฟล์ ไม่แยกโฟลเดอร์ต่อกล้อง เพราะฝั่งอ่านตัวเดียวก็วนคิวรวมได้
    ไฟล์เก่าที่ถ่ายก่อนมีกล้อง 2 ตัวจะขึ้นต้นด้วยปี = UNKNOWN

    ข้ามท่อนที่เป็นตัวเลขล้วน เพราะ move_to_sent() เติม timestamp นำหน้าตอนย้ายเข้า send/
    (20260818_101500_IN_...jpg) ถ้าดูแค่ท่อนแรกรูปใน send/ จะกลายเป็น UNKNOWN ทั้งโฟลเดอร์
    แล้ว --batch send/ จะรวม IN กับ OUT เป็นกล้องเดียว = ผลทดสอบเชื่อไม่ได้
    """
    head = next((t for t in img_path.stem.split("_") if not t.isdigit()), "")
    return head.upper() if head else "UNKNOWN"


def best_plate(plates):
    """เลือกป้ายที่ conf สูงสุด — ข้าม crop ว่าง (bbox ที่ติดขอบภาพจนกว้าง/สูงเป็น 0)"""
    usable = [p for p in plates if p["image"].size > 0]
    return max(usable, key=lambda p: p["conf"]) if usable else None


def append_log(path: Path, msg: str):
    """ต่อท้าย log พร้อมเวลา — เขียนไม่ได้ก็ข้าม ไม่ให้ล้มทั้งลูป"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    except OSError as e:
        print(f"          ⚠️  เขียน {path} ไม่ได้: {e}")


def log_yolo_hit(img_path: Path, cars: int, plates: int, conf: float):
    """ต่อท้าย log ตอน YOLO เจอป้าย"""
    append_log(
        YOLO_LOG,
        f"{img_path.name} cam={camera_of(img_path)} "
        f"car={cars} plate={plates} conf={conf:.2f}",
    )


def log_failed(img_path: Path, plate: str, reason: str):
    """log ตอนส่งไม่สำเร็จ — ไฟล์เดิมเขียนครั้งเดียวต่อการรัน ไม่ใช่ทุก 3 วิที่ retry

    ponytail: dedupe ในหน่วยความจำ (รีสตาร์ทแล้วเริ่มนับใหม่) — เน็ตล่มทั้งคืนแล้วคิว 200 ใบ
    วน retry ทุก POLL_INTERVAL จะได้ log แสนบรรทัด ถ้าอยากรู้ว่า retry ไปกี่รอบค่อยถอดออก
    """
    if str(img_path) in _failed_logged:
        return
    _failed_logged.add(str(img_path))
    append_log(
        FAILED_LOG,
        f"{img_path.name} cam={camera_of(img_path)} plate={plate} — {reason}",
    )


def read_plate_of(img_path: Path):
    """คืน (plate, province, det_conf, ocr_score, prov_score) ของภาพหนึ่งใบ — อ่านไม่ได้คืน UNKNOWN

    ocr_score คือน้ำหนักที่ vote_group ใช้ถ่วงเสียงของเฟรมนี้
    """
    import cv2

    detect, read_plate = pipeline()

    frame = cv2.imread(str(img_path))
    if frame is None:
        # ponytail: ไฟล์เสีย = ถือว่าเฟรมนี้อ่านป้ายไม่ได้ ไม่ raise เพราะจะพาทั้งก้อนตกไป
        # อยู่ในมือ except ของ main() แล้วค้างใน unsend/ พังซ้ำทุกรอบตลอดไป
        print(f"          ⚠️ {img_path.name} ไฟล์เสีย/อ่านไม่ได้ — ข้ามเฟรมนี้")
        return "UNKNOWN", "UNKNOWN", 0.0, 0.0, 0.0

    cars, plates = detect(frame, conf=DETECT_CONF)
    plate = best_plate(plates)
    if plate is None:
        print(
            f"          🔍 {img_path.name} YOLO เห็น car={len(cars)} plate={len(plates)}"
        )
        return "UNKNOWN", "UNKNOWN", 0.0, 0.0, 0.0

    log_yolo_hit(img_path, len(cars), len(plates), plate["conf"])

    license_id, province, score, prov_score = read_plate(
        plate["image"], SPLIT_RATIO, PROVINCE_THRESHOLD, OCR_CONF
    )
    return (
        license_id or "UNKNOWN",
        province or "UNKNOWN",
        plate["conf"],
        score,
        prov_score,
    )


def group_frames(images):
    """แบ่งรูป (เรียงเก่า→ใหม่มาแล้ว) เป็นก้อนละคัน — คืน list ของ list

    ponytail: ไม่มี tracker ข้ามไฟล์ (ByteTrack ต้องใช้ lap ซึ่งลงบน Pi OS ไม่ได้ PEP 668)
    เลยใช้ "กล้องเดิม + ถ่ายติดๆ กัน" แทน track id — ฝั่งถ่ายเซฟทุก 0.5 วิเฉพาะตอนมีการ
    เคลื่อนไหว ช่องว่างเกิน GROUP_GAP จึงแปลว่ารถคันก่อนหน้าออกจากเฟรมไปแล้ว

    ก้อนโตเกิน MAX_GROUP ต้องตัดเป็นก้อนใหม่ ไม่ใช่โตต่อไปเรื่อยๆ — ฝั่งถ่าย LINGER=3 วิ
    ทำให้ตอนรถต่อคิวกันมา เฟรมไม่มีช่องว่างเกิน GROUP_GAP เลยแม้แต่ครั้งเดียว ถ้าไม่ตัด
    รถ 5 คันจะรวมเป็นก้อนเดียว แล้ว vote_group อ่านแค่เฟรมท้าย = ได้ป้ายคันสุดท้ายคันเดียว
    """
    groups, open_group = [], {}  # open_group: กล้อง → ก้อนที่ยังรับเฟรมอยู่

    for img in images:
        cam, mtime = camera_of(img), img.stat().st_mtime
        g = open_group.get(cam)
        if (
            g is not None
            and len(g) < MAX_GROUP
            and mtime - g[-1].stat().st_mtime <= GROUP_GAP
        ):
            g.append(img)
        else:
            open_group[cam] = [img]
            groups.append(open_group[cam])

    return groups


def group_closed(group, now) -> bool:
    """ก้อนนี้ปิดรับเฟรมแล้วหรือยัง — ยังไม่ปิดก็รอไว้ก่อน เดี๋ยวเฟรมถัดไปของคันนี้ตามมา"""
    return len(group) >= MAX_GROUP or now - group[-1].stat().st_mtime > GROUP_GAP


def vote_side(cam: str) -> str:
    """ฝั่งของก้อนที่กล้องนี้ควรหยิบเฟรมมาอ่าน — env ทับ VOTE_FROM ได้"""
    side = os.environ.get(f"VOTE_FROM_{cam}") or VOTE_FROM.get(cam, VOTE_FROM_DEFAULT)
    return side.strip().lower()


def pick_frames(group, n):
    """หยิบ n เฟรมตามทิศทางของกล้องนี้ — คืน (ชุดแรก, ชุดสำรองที่ลึกเข้าไปอีก n เฟรม)

    ponytail: ทิศทางมาจาก VOTE_FROM ไม่ใช่เดาจากรูป — จะรู้ว่าฝั่งไหนดีต้องรัน YOLO
    ทุกเฟรมในก้อนก่อน ซึ่งเป็นส่วนที่แพงที่สุดของไปป์ไลน์ และคำตอบก็เหมือนเดิมทุกคัน
    เพราะกล้องไม่ได้ขยับ ตั้งครั้งเดียวตอนติดกล้องแล้วจบ
    """
    if len(group) <= n:
        return list(group), []
    if vote_side(camera_of(group[0])) == "head":
        return group[:n], group[n : 2 * n]
    return group[-n:], group[-2 * n : -n]


def vote_group(group):
    """โหวตป้ายจากหลายเฟรมของรถคันเดียวกัน — คืน (รูปที่ดีที่สุด, plate, province, conf, คะแนนโหวต, conf ป้าย, conf จังหวัด)

    น้ำหนักของแต่ละเฟรม = det_conf × ocr_score (ป้ายชัดและอ่านมั่นใจถึงจะมีสิทธิ์เต็มเสียง)
    รูปที่ดีที่สุด = เฟรมที่โหวตให้ป้ายที่ชนะด้วยน้ำหนักสูงสุด — คือใบที่ส่งขึ้น backend
    """
    from ocr import plate_vote, province_vote

    def read_frames(frames):
        out = []
        for img in frames:
            if str(img) not in _ocr_cache:
                _ocr_cache[str(img)] = read_plate_of(img)
            plate, province, det, score, prov = _ocr_cache[str(img)]
            if plate != "UNKNOWN":
                out.append((img, plate, province, det, det * score, score, prov))
        return out

    picked, backup = pick_frames(group, VOTE_N)
    reads = read_frames(picked)

    # อ่านไม่ออกทั้งชุด แต่ YOLO เห็นกล่องป้ายแล้วจริง (det_conf > 0) = มีรถแน่ ค่อยจ่าย
    # OCR ชุดถัดไปในทิศเดิม ถ้าไม่เห็นป้ายเลยแปลว่าไม่มีรถ (ใบไม้ไหว/แสงเปลี่ยน) จ่ายไป
    # ก็ได้ UNKNOWN เหมือนเดิม — เคสนี้เกิดบ่อยกว่าเคสรถจริงหลายเท่า
    saw_plate = any(_ocr_cache[str(i)][2] > 0 for i in picked)
    if not reads and saw_plate:
        reads = read_frames(backup)

    if not reads:
        return None, "UNKNOWN", "UNKNOWN", 0.0, 0.0, 0.0, 0.0

    plate, votes = plate_vote([(r[1], r[4]) for r in reads])
    province = province_vote([r[2] for r in reads if r[2] != "UNKNOWN"])

    agree = [
        r for r in reads if r[1] == plate
    ] or reads  # ชนะแบบประกอบตัวอักษรใหม่ = ไม่มีเฟรมไหนตรง
    best = max(agree, key=lambda r: r[4])
    # conf จังหวัด = เฟรมที่มั่นใจที่สุดในบรรดาเฟรมที่โหวตให้จังหวัดที่ชนะ — best เลือกมา
    # ด้วยน้ำหนักของ "ป้าย" จังหวัดของมันจึงไม่จำเป็นต้องเป็นตัวที่ชนะโหวต
    prov_conf = max((r[6] for r in reads if r[2] == province), default=0.0)
    return (
        best[0],
        plate or "UNKNOWN",
        province or "UNKNOWN",
        best[3],
        votes,
        best[5],
        prov_conf,
    )


def is_duplicate(cam: str, plate: str, captured: datetime) -> bool:
    """ยิงป้ายนี้จากกล้องนี้ไปแล้วภายใน POST_COOLDOWN วิ = รถคันเดิม

    กล้องหน้าไม้กั้น (linger ยาว) ถ่ายรถที่จอดรอได้เกิน MAX_GROUP เฟรม ฝั่งนี้จึงตัดเป็น
    หลายก้อน แต่ละก้อนโหวตได้ป้ายเดียวกัน แล้วยิงซ้ำ 2-3 ครั้งต่อรถหนึ่งคัน

    abs() เพราะ main() ไล่ก้อนใหม่ก่อน ก้อนเก่าของรถคันเดียวกันจึงมาถึงทีหลัง
    ponytail: เทียบเวลาที่ถ่าย ไม่ใช่เวลาที่ยิง — เน็ตล่มแล้วคิวค้างเป็นชั่วโมง ก้อนของรถ
    คันเดียวกันจะถูกยิงห่างกันไม่กี่วิ ทั้งที่ถ่ายมาพร้อมกัน ถ้าเทียบเวลายิงจะกันไม่ได้
    """
    prev = _last_post.get((cam, plate))
    return prev is not None and abs((captured - prev).total_seconds()) <= POST_COOLDOWN


def post_detection(
    img_path: Path,
    plate: str,
    province: str,
    conf: float,
    plate_conf: float = 0.0,
    prov_conf: float = 0.0,
) -> bool:
    """ยิงขึ้น backend คืน True ถ้าสำเร็จ"""
    import requests

    now = datetime.now()
    # เวลาที่ถ่าย ≠ เวลาที่ส่ง — คิวค้างเป็นชั่วโมงแล้วยังต้องได้เวลาถ่ายที่ถูก
    captured = datetime.fromtimestamp(img_path.stat().st_mtime)

    cam = camera_of(img_path)
    if is_duplicate(cam, plate, captured):
        # คืน True = จัดการก้อนนี้จบแล้ว ย้ายเข้า send/ ไปเลย ไม่ต้องยิง
        print(
            f"[{now:%H:%M:%S}] ⏭️ {img_path.name} cam={cam} plate={plate} ยิงไปแล้ว — ข้าม"
        )
        return True

    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    payload = {
        "image": b64,
        "plate": plate,
        "province": province,
        "confidence": conf,
        "plate_confidence": plate_conf,
        "province_confidence": prov_conf,
        "captured_at": captured.isoformat(),
        "camera": cam,
    }

    try:
        res = requests.post(API_URL, json=payload, timeout=15)
    except requests.RequestException as e:
        # เน็ตหลุด/backend ล่ม = คืน False ไม่ใช่ raise — ไฟล์ค้างไว้รอบหน้าลองใหม่
        # ถ้าปล่อยให้ raise จะไปโดน except ของ main() ซึ่งตีความว่า "ข้อมูลเสีย" แล้วทิ้ง
        print(f"[{now:%H:%M:%S}] 🌐 {img_path.name} ส่งไม่ได้: {e} — รอบหน้าลองใหม่")
        log_failed(img_path, plate, f"เน็ต/backend: {type(e).__name__}: {e}")
        return False

    if res.status_code == 201:
        _last_post[(cam, plate)] = captured
        for k, t in list(_last_post.items()):  # ทิ้งของเก่า กันโตไม่จำกัดตอนรันทั้งวัน
            if abs((captured - t).total_seconds()) > POST_COOLDOWN:
                del _last_post[k]
        print(
            f"[{now:%H:%M:%S}] ✅ ส่ง {img_path.name} cam={cam} "
            f"plate={plate} province={province}"
        )
        return True

    print(
        f"[{now:%H:%M:%S}] ❌ {img_path.name} ล้มเหลว {res.status_code}: {res.text[:120]}"
    )
    log_failed(img_path, plate, f"HTTP {res.status_code}: {res.text[:120]}")
    return False


def send_group(group) -> bool:
    """โหวตทั้งก้อนแล้วส่งรูปที่ดีที่สุดใบเดียว คืน True ถ้าจัดการก้อนนี้จบแล้ว"""
    best, plate, province, conf, votes, plate_conf, prov_conf = vote_group(group)
    if best is None:
        # ponytail: คืน True เพื่อให้ย้ายเข้า send/ ไปเลย ไม่งั้นวนอ่านไฟล์เดิมทุกรอบ
        print(f"          ⏭️ {group[-1].name} +{len(group) - 1} เฟรม ไม่เจอป้าย — ไม่ส่ง")
        # log ด้วย: นี่คือ "รถผ่านแต่ไม่ได้ตรวจเจอ" ซึ่งเงียบที่สุดในบรรดาทางที่รถหลุด
        log_failed(group[-1], "UNKNOWN", f"อ่านป้ายไม่ออกทั้ง {len(group)} เฟรม")
        return True

    print(f"          🗳️ โหวตจาก {len(group)} เฟรม → {plate} (คะแนน {votes:.2f})")
    return post_detection(best, plate, province, conf, plate_conf, prov_conf)


def move_to_sent(img_path: Path):
    """ย้ายไฟล์ที่ส่งแล้วไป send/ — เติม timestamp กันชื่อชน"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = SENT_DIR / f"{ts}_{img_path.name}"
    try:
        shutil.move(str(img_path), str(dest))
    except FileNotFoundError:
        # ฝั่งถ่าย make_room() ลบไฟล์นี้ทิ้งไปแล้ว (คนละโปรเซส คิวเต็มพร้อมกัน)
        # ปล่อยผ่าน — ไม่ใช่เหตุให้ทั้งลูปตาย
        return
    print(f"          ย้ายไป {dest.name}")
    prune(SENT_DIR, MAX_SENT_FILES)


def images_in(d: Path):
    """รูปในโฟลเดอร์ เรียงเก่า→ใหม่"""
    return sorted(
        [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.stat().st_mtime,
    )


def prune(d: Path, keep: int):
    """ลบรูปเก่าสุดในโฟลเดอร์ให้เหลือไม่เกิน keep รูป"""
    files = images_in(d)
    dropped = files[: len(files) - keep]
    for old in dropped:
        old.unlink()
        print(f"          🗑️ ลบ {d.name}/{old.name} (เกิน {keep} รูป)")

    # ทิ้งจากคิวรอส่ง = รูปที่ไม่เคยผ่าน OCR เลย นี่คือรถที่หายแบบไม่มีร่องรอยที่สุด
    # log บรรทัดเดียวต่อรอบ ไม่ใช่ต่อไฟล์ — คิวล้นทีนึงหลายสิบใบ
    if dropped and d == UNSEND_DIR:
        append_log(
            FAILED_LOG,
            f"ทิ้งจากคิว {len(dropped)} รูป (ยังไม่เคยอ่าน) "
            f"เก่าสุด={dropped[0].name} ใหม่สุด={dropped[-1].name}",
        )


def main():
    UNSEND_DIR.mkdir(parents=True, exist_ok=True)
    SENT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"เริ่มเฝ้า {UNSEND_DIR} (เช็คทุก {POLL_INTERVAL} วิ) — Ctrl+C เพื่อหยุด")

    while True:
        prune(UNSEND_DIR, MAX_UNSEND_FILES)  # คิวล้นก่อนอ่าน = ทิ้งของเก่าทันที
        images = images_in(UNSEND_DIR)
        alive = {str(i) for i in images}
        for gone in set(_ocr_cache) - alive:  # ส่งไปแล้ว/โดน prune
            del _ocr_cache[gone]
        _failed_logged.intersection_update(alive)  # ไฟล์หายไปแล้วไม่ต้องจำว่า log แล้ว

        now = time.time()
        groups = [g for g in group_frames(images) if group_closed(g, now)]

        # ponytail: ใหม่→เก่า เพราะกล้อง 2 ตัวป้อนเร็วกว่าที่ฝั่งนี้อ่านทัน ถ้าไล่จากเก่าสุด
        # รถที่เพิ่งผ่านต้องรอจนกว่าจะเคลียร์คิวหมด (คิว 200 ใบ ≈ 3 นาที) ของเก่าตกค้างก็
        # ปล่อยให้ prune ทิ้งไป — ป้ายรถเมื่อ 3 นาทีที่แล้วไม่มีใครรอดูแล้ว
        progressed = False
        for g in reversed(groups):
            try:
                done = send_group(g)
                # ถ้าส่งไม่สำเร็จ (HTTP ไม่ 201 / เน็ตหลุด) ปล่อยไฟล์ไว้ที่เดิม รอบหน้าลองใหม่
            except Exception as e:
                # ข้อมูลเสีย/บั๊ก — ย้ายออกจากคิวเลย ไม่งั้นก้อนนี้พังซ้ำทุกรอบตลอดไป
                print(f"          ⚠️ error {g[-1].name}: {e} — ย้ายออกจากคิว")
                log_failed(g[-1], "?", f"ประมวลผลพัง {type(e).__name__}: {e}")
                done = True
            if done:
                for img in g:  # ทั้งก้อนจบพร้อมกัน เหลือค้างไว้จะโดนโหวตซ้ำรอบหน้า
                    move_to_sent(img)
                progressed = True

        # นอนเมื่อรอบนี้ไม่คืบหน้า — ทั้งกรณียังไม่มีก้อนไหนปิด และกรณีส่งไม่ผ่านทุกก้อน
        # (ไม่งั้นวนซ้ำเต็มสปีดกินซีพียู/ถล่ม backend)
        if not progressed:
            time.sleep(POLL_INTERVAL)


def load_truth(path: Path):
    """อ่านเฉลย `ชื่อไฟล์,ป้าย,จังหวัด` → dict

    ใส่เฉพาะคันที่อยากวัดก็ได้ ไม่ต้องครบทุกรูป — และเพราะทุกเฟรมในก้อนเดียวกันคือรถคันเดียวกัน
    จึงพิมพ์เฉลยแค่ "เฟรมเดียวต่อคัน" พอ ที่เหลือ score_group() หาให้เอง

    ป้ายว่าง = ไม่มีรถ/ไม่ควรอ่านออก (ใช้จับ false positive) จังหวัดว่าง = ไม่วัดจังหวัดคันนี้

    ponytail: utf-8-sig เพราะ Excel เซฟ CSV ภาษาไทยมาพร้อม BOM เสมอ อ่านด้วย utf-8 เฉยๆ
    ชื่อไฟล์แถวแรกจะมีตัว BOM (U+FEFF) ติดหัว แล้ว lookup พลาดเงียบๆ ทั้งไฟล์ — หาสาเหตุยากมาก
    """
    truth = {}

    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if row[0].strip().lower() in ("filename", "file", "ชื่อไฟล์"):  # หัวตาราง
                continue

            cell = (row + ["", ""])[:3]
            truth[cell[0].strip()] = (cell[1].strip(), cell[2].strip())

    return truth


def score_group(group, plate, province, truth):
    """เทียบผลโหวตกับเฉลย → (ป้ายถูกไหม, จังหวัดถูกไหม) — None = ไม่มีเฉลย ไม่นับ"""
    want = next((truth[f.name] for f in group if f.name in truth), None)
    if want is None:
        return None, None

    want_plate, want_province = want
    got_plate = "" if plate == "UNKNOWN" else plate
    got_province = "" if province == "UNKNOWN" else province

    return (
        got_plate == want_plate,
        got_province == want_province if want_province else None,
    )


def batch(folder: Path, only_cam: str = None, truth_path: Path = None):
    """รันทั้งโฟลเดอร์ ไม่ยิง backend — เทียบอัตราอ่านออกแยกตามกล้อง

    ใช้ตอบว่า "โมเดลแย่" หรือ "รูปไม่เคยมาถึงโมเดล" — ชี้ไปที่ send/ แล้วดูเลขท้าย
    ถ้า IN กับ OUT ต่างกันมาก = ปัญหาอยู่ที่กล้อง/มุม ไม่ใช่โมเดล

    only_cam = ดูกล้องเดียว (เช่น IN) บรรทัด "อ่านแล้ว" ใต้แต่ละคันคือเฟรมที่ vote_group
    ยอมจ่าย OCR จริงๆ — ถ้าเฟรมที่อ่านออกอยู่ต้นก้อนแต่โหวตไปหยิบแต่เฟรมท้าย แปลว่า
    VOTE_N เลือกผิดฝั่งสำหรับกล้องตัวนั้น (รถวิ่งหนีกล้อง ป้ายเล็กลงเรื่อยๆ)
    """
    from collections import Counter

    truth = load_truth(truth_path) if truth_path else {}

    total, hit, sc = Counter(), Counter(), Counter()
    wrong = []
    for g in group_frames(images_in(folder)):  # นับเป็นคัน ไม่ใช่เป็นรูป — ให้ตรงกับตอนรันจริง
        cam = camera_of(g[0])
        if only_cam and cam != only_cam:
            continue
        total[cam] += 1
        try:
            best, plate, province, conf, votes, *_ = vote_group(g)
        except Exception as e:
            print(f"⚠️ {g[-1].name}: {e}")
            continue
        if plate != "UNKNOWN":
            hit[cam] += 1
        name = best.name if best else g[-1].name
        print(
            f"{name} ({len(g)} เฟรม) cam={cam} plate={plate} "
            f"province={province} conf={conf:.2f} vote={votes:.2f}"
        )
        for i, f in enumerate(g):
            r = _ocr_cache.get(str(f))
            if r:  # ไม่มีใน cache = โหวตไม่ได้อ่านเฟรมนี้เลย
                print(
                    f"    อ่านแล้ว #{i + 1}/{len(g)} {f.name} "
                    f"det={r[2]:.2f} ocr={r[3]:.2f} → {r[0]}"
                )

        if truth:
            plate_ok, province_ok = score_group(g, plate, province, truth)

            if plate_ok is None:
                sc[cam, "ไม่มีเฉลย"] += 1
            else:
                want_plate, want_province = next(
                    truth[f.name] for f in g if f.name in truth
                )
                sc[cam, "n"] += 1
                sc[cam, "ป้ายถูก"] += plate_ok

                if not plate_ok:
                    wrong.append(
                        f"{name} ป้าย ได้ {plate} เฉลย {want_plate or '(ไม่มีรถ)'}"
                    )

                if province_ok is not None:
                    sc[cam, "จว.n"] += 1
                    sc[cam, "จว.ถูก"] += province_ok

                    if not province_ok:
                        wrong.append(f"{name} จังหวัด ได้ {province} เฉลย {want_province}")

    print("\n===== สรุป =====")
    for cam in sorted(total):
        n, h = total[cam], hit[cam]
        print(f"{cam}: อ่านป้ายได้ {h}/{n} คัน ({h / n * 100:.0f}%)")

    if not truth:
        return

    # "อ่านออก" ข้างบนนับแค่ว่าไม่ใช่ UNKNOWN — ป้ายที่ซ่อมไม่ได้ก็ยังนับว่าอ่านออก
    # ท่อนนี้ต่างหากที่ตอบว่า "ถูก" เลขที่เอาไปเทียบก่อน/หลังแก้โค้ดคือเลขนี้
    print()
    print("===== เทียบกับเฉลย =====")

    for line in wrong:
        print(f"  ❌ {line}")

    for cam in sorted(total):
        n = sc[cam, "n"]
        if not n:
            continue

        ok = sc[cam, "ป้ายถูก"]
        line = f"{cam}: ป้ายถูก {ok}/{n} ({ok / n * 100:.0f}%)"

        pn = sc[cam, "จว.n"]
        if pn:
            po = sc[cam, "จว.ถูก"]
            line += f" | จังหวัดถูก {po}/{pn} ({po / pn * 100:.0f}%)"

        skipped = sc[cam, "ไม่มีเฉลย"]
        if skipped:
            line += f" | ไม่มีเฉลย {skipped} คัน (ไม่นับ)"

        print(line)


def lag(folder: Path = None):
    """ฝั่งอ่านตามหลังฝั่งถ่ายกี่วินาที — ตอบว่า Pi แบ่งงานทันหรือไม่

    ชื่อไฟล์ใน send/ มีเวลาที่ move_to_sent() ย้ายเข้ามา (= อ่านเสร็จแล้ว) นำหน้า
    ส่วน mtime คือเวลาที่ถ่าย ผลต่าง = คิวหน่วงเท่าไหร่

    อ่านผล: ตัวเลขไต่ขึ้นเรื่อยๆ = ตามไม่ทันจริง งานล้นเข้ามาเร็วกว่าที่อ่านออก
    คงที่ = ทัน แค่มีหน่วงประจำ | ตัวเลขต่ำแต่ของหายไปเยอะ = โดน prune ทิ้งก่อนถึงโมเดล
    เทียบ n ของ IN กับ OUT ด้วย — ต่างกันมาก = กล้องตัวหนึ่งกินคิวไปหมด
    """
    from collections import defaultdict
    from statistics import median

    rows = []
    for p in (folder or SENT_DIR).glob("*.jpg"):
        head = p.stem.split("_")
        if len(head) < 2 or not (head[0].isdigit() and head[1].isdigit()):
            continue  # ยังไม่ผ่าน move_to_sent = ไม่มี timestamp นำหน้าให้เทียบ
        try:
            done = datetime.strptime(head[0] + head[1], "%Y%m%d%H%M%S")
        except ValueError:
            continue
        shot = datetime.fromtimestamp(p.stat().st_mtime)
        rows.append((shot, (done - shot).total_seconds(), camera_of(p)))

    if not rows:
        print(f"ไม่มีไฟล์ที่เทียบเวลาได้ใน {folder or SENT_DIR}")
        return

    rows.sort()
    for shot, secs, cam in rows[-30:]:
        print(f"{shot:%H:%M:%S} {cam:<7} ช้า {secs:6.1f} วิ")

    per_cam = defaultdict(list)
    for _, secs, cam in rows:
        per_cam[cam].append(secs)

    print("\n===== สรุป =====")
    for cam in sorted(per_cam):
        v = per_cam[cam]
        print(f"{cam}: {len(v)} รูป | หน่วงกลาง {median(v):.1f} วิ | แย่สุด {max(v):.1f} วิ")

    span = (rows[-1][0] - rows[0][0]).total_seconds()
    if span > 0:
        print(f"ช่วงเวลาที่เก็บได้ {span / 60:.1f} นาที = {len(rows) / span:.1f} รูป/วิ")
    # คิวเต็มค้าง = อ่านไม่ทันแน่นอน (prune กำลังทิ้งของก่อนถึงโมเดล)
    q = len(images_in(UNSEND_DIR)) if UNSEND_DIR.is_dir() else 0
    print(f"คิว unsend/ ตอนนี้ {q}/{MAX_UNSEND_FILES} รูป")
    return rows


def _selfcheck():
    """python3 Run_v2.py --check — เช็คตรรกะล้วน ไม่แตะโมเดล/กล้อง/backend"""
    import tempfile
    from types import SimpleNamespace

    import numpy as np

    ok = np.ones((10, 10, 3), dtype=np.uint8)
    empty = np.empty((0, 0, 3), dtype=np.uint8)

    assert best_plate([]) is None
    assert best_plate([{"image": empty, "conf": 0.9}]) is None
    picked = best_plate([{"image": ok, "conf": 0.4}, {"image": ok, "conf": 0.8}])
    assert picked["conf"] == 0.8

    # ไม่เจอป้าย = ห้ามยิง backend
    g = globals()
    g["FAILED_LOG"] = Path(tempfile.gettempdir()) / "check_filed_to_send.txt"
    g["read_plate_of"] = lambda p: ("UNKNOWN", "UNKNOWN", 0.0, 0.0, 0.0)
    g["post_detection"] = lambda *a: (_ for _ in ()).throw(AssertionError("ไม่ควรส่ง"))
    assert send_group([Path("ไม่มีจริง.jpg")]) is True

    # OUT = head: ชุดแรก "เห็นป้ายแต่อ่านตัวไม่ออก" (det>0) → เลื่อนไปอ่านชุดถัดไปในทิศเดิม
    front = [Path(f"OUT_{i}.jpg") for i in range(VOTE_N)]
    back = [Path(f"OUT_late{i}.jpg") for i in range(VOTE_N)]
    g["read_plate_of"] = lambda p: (
        ("1กข1234", "ชลบุรี", 0.8, 0.9, 0.9)
        if "late" in p.name
        else ("UNKNOWN", "UNKNOWN", 0.6, 0.0, 0.0)
    )
    _ocr_cache.clear()
    best, plate, *_ = vote_group(front + back)
    assert plate == "1กข1234" and best in back, (plate, best)

    # แต่ถ้า YOLO ไม่เห็นป้ายเลย (det=0 = ไม่มีรถในเฟรม) ห้ามถอยไปอ่านเพิ่ม — เสีย OCR เปล่า
    _ocr_cache.clear()
    read_count = [0]

    def count_read(p):
        read_count[0] += 1
        return ("UNKNOWN", "UNKNOWN", 0.0, 0.0, 0.0)

    g["read_plate_of"] = count_read
    assert vote_group(front + back)[0] is None
    assert read_count[0] == VOTE_N, f"อ่านเกินชุดที่หยิบไป {read_count[0]} เฟรม"
    _ocr_cache.clear()

    # เคสจริง 27 ส.ค. 69: รถวิ่งลอดใต้กล้อง IN ป้ายอยู่ใน 2 เฟรมแรกของก้อน 12 เฟรม
    # ที่เหลือรถเลยกล้องไปแล้ว (det=0) — ตอน IN ยังเป็น tail คันนี้หายทั้งคัน
    car = [Path(f"IN_{i}.jpg") for i in range(MAX_GROUP)]
    g["read_plate_of"] = lambda p: (
        ("1ขท8669", "ขอนแก่น", 0.9, 0.9, 0.9)
        if int(p.stem.split("_")[1]) < 2
        else ("UNKNOWN", "UNKNOWN", 0.0, 0.0, 0.0)
    )
    _ocr_cache.clear()
    assert vote_group(car)[1] == "1ขท8669", "IN=head ป้ายอยู่ต้นก้อนต้องอ่านเจอ"
    _ocr_cache.clear()

    # ทิศทางมาจาก VOTE_FROM ต่อกล้อง (ตอนนี้ head ทั้งคู่) และ env ต้องทับได้
    ins = [Path(f"IN_{i}.jpg") for i in range(MAX_GROUP)]
    outs = [Path(f"OUT_{i}.jpg") for i in range(MAX_GROUP)]
    assert VOTE_FROM == {"IN": "head", "OUT": "head"}, VOTE_FROM  # เทสต์ล่างผูกกับค่านี้
    assert pick_frames(ins, 5) == (ins[:5], ins[5:10]), "IN=head ต้องหยิบจากต้นก้อน"
    assert pick_frames(outs, 5) == (outs[:5], outs[5:10]), "OUT=head ต้องหยิบจากต้นก้อน"
    assert pick_frames(ins[:3], 5) == (ins[:3], []), "ก้อนเล็กกว่า n = อ่านทั้งก้อน"
    assert vote_side("UNKNOWN") == VOTE_FROM_DEFAULT, "กล้องที่ไม่ได้ตั้งไว้"
    os.environ["VOTE_FROM_IN"] = "TAIL "  # เผื่อพิมพ์ตัวใหญ่/ติดช่องว่างมา
    assert pick_frames(ins, 5) == (ins[-5:], ins[-10:-5]), "env ต้องทับ VOTE_FROM"
    del os.environ["VOTE_FROM_IN"]
    _ocr_cache.clear()

    # โหวตข้ามเฟรม: 2 เฟรมอ่านได้ 1กข1234, อีกเฟรมอ่านมั่วแต่ conf สูงกว่า → เสียงส่วนใหญ่ชนะ
    # และรูปที่ส่งต้องเป็นเฟรมที่โหวตให้ผู้ชนะด้วยน้ำหนักสูงสุด ไม่ใช่เฟรมที่ conf สูงสุดในก้อน
    reads = {
        "a.jpg": ("1กข1234", "ชลบุรี", 0.7, 0.7, 0.6),
        "b.jpg": ("1กข1234", "ชลบุรี", 0.9, 0.8, 0.95),  # หนักสุดในบรรดาเฟรมที่ตรงกับผู้ชนะ
        "c.jpg": ("9ฮฮ9999", "ระยอง", 0.95, 0.99, 0.99),
    }
    g["read_plate_of"] = lambda p: reads[p.name]
    _ocr_cache.clear()
    best, plate, province, conf, votes, plate_conf, prov_conf = vote_group(
        [Path(n) for n in reads]
    )
    assert plate == "1กข1234", plate
    assert best.name == "b.jpg", best
    assert province == "ชลบุรี" and conf == 0.9 and votes > 0
    # conf ที่ยิงขึ้น backend ต้องมาจากเฟรมที่ชนะโหวต ไม่ใช่เฟรมที่ conf สูงสุดในก้อน (c.jpg)
    assert plate_conf == 0.8, plate_conf
    assert prov_conf == 0.95, prov_conf
    _ocr_cache.clear()

    # ---- เฉลย: อ่าน CSV + เทียบผลโหวต ----
    truth_csv = Path(tempfile.gettempdir()) / "check_truth.csv"
    truth_csv.write_text(
        "filename,plate,province\n"  # หัวตารางต้องถูกข้าม
        "# คอมเมนต์ต้องถูกข้าม\n"
        "\n"  # บรรทัดว่างต้องถูกข้าม
        "IN_a.jpg,1กข1234,ชลบุรี\n"
        "IN_c.jpg,2ขค5678\n"  # ไม่ใส่จังหวัด = ไม่วัดจังหวัดคันนี้
        "IN_d.jpg,,\n",  # ป้ายว่าง = ไม่ควรมีรถ ใช้จับ false positive
        encoding="utf-8-sig",  # Excel เซฟมาแบบนี้ ต้องอ่านออก
    )
    truth = load_truth(truth_csv)
    assert set(truth) == {"IN_a.jpg", "IN_c.jpg", "IN_d.jpg"}, truth
    assert truth["IN_a.jpg"] == ("1กข1234", "ชลบุรี")
    assert truth["IN_c.jpg"] == ("2ขค5678", ""), "คอลัมน์ขาดต้องเป็นค่าว่าง ไม่ใช่ IndexError"

    # เฉลยเฟรมเดียวพอ — ทุกเฟรมในก้อนคือรถคันเดียวกัน ไม่ต้องพิมพ์เฉลยทุกรูป
    grp = [Path("IN_a.jpg"), Path("IN_b.jpg")]
    assert score_group(grp, "1กข1234", "ชลบุรี", truth) == (True, True)
    assert score_group(grp, "1กข1234", "ระยอง", truth) == (True, False)
    assert score_group(grp, "9ฮฮ9999", "ชลบุรี", truth) == (False, True)

    # ไม่มีเฉลยในก้อนนี้เลย = ไม่นับ ไม่ใช่นับว่าผิด (ให้ label แค่บางคันได้)
    assert score_group([Path("IN_zzz.jpg")], "1กข1234", "ชลบุรี", truth) == (None, None)

    # เฉลยไม่มีจังหวัด = ข้ามการวัดจังหวัด แต่ยังวัดป้าย
    assert score_group([Path("IN_c.jpg")], "2ขค5678", "ระยอง", truth) == (True, None)

    # UNKNOWN ต้องถือเป็น "อ่านไม่ออก" ไม่ใช่สตริงที่เอาไปเทียบตรงๆ
    assert score_group(grp, "UNKNOWN", "UNKNOWN", truth) == (False, False)
    # ไม่มีรถแล้วอ่านไม่ออก = ถูก / ไม่มีรถแต่ดันอ่านออก = false positive ต้องจับได้
    assert score_group([Path("IN_d.jpg")], "UNKNOWN", "UNKNOWN", truth)[0] is True
    assert score_group([Path("IN_d.jpg")], "1กข1234", "UNKNOWN", truth)[0] is False
    truth_csv.unlink()

    # กันยิงซ้ำ: รถจอดรอไม้กั้นนานจนถูกตัดเป็นหลายก้อน ต้องขึ้น backend ครั้งเดียว
    t0 = datetime(2026, 8, 19, 8, 0, 0)
    _last_post.clear()
    assert not is_duplicate("IN", "1กข1234", t0), "ยังไม่เคยยิง ต้องไม่ใช่ซ้ำ"
    _last_post[("IN", "1กข1234")] = t0
    assert is_duplicate("IN", "1กข1234", t0 + timedelta(seconds=POST_COOLDOWN - 1))
    assert is_duplicate("IN", "1กข1234", t0 - timedelta(seconds=5)), "ก้อนเก่ามาทีหลัง"
    assert not is_duplicate("IN", "1กข1234", t0 + timedelta(seconds=POST_COOLDOWN + 1))
    assert not is_duplicate("OUT", "1กข1234", t0), "คนละกล้อง = คนละเหตุการณ์"
    assert not is_duplicate("IN", "9ฮฮ9999", t0), "คนละป้าย = คนละคัน"
    _last_post.clear()

    # pk กล้องอ่านจาก prefix ชื่อไฟล์ที่ Run_v2_capture.py เซฟมา
    assert camera_of(Path("IN_20260812_101500_123.jpg")) == "IN"
    assert camera_of(Path("OUT_20260812_101500_123.jpg")) == "OUT"
    assert camera_of(Path("20260812_101500_123.jpg")) == "UNKNOWN"  # ไฟล์เก่าก่อนมี 2 กล้อง
    # ชื่อแบบที่อยู่ใน send/ (move_to_sent เติม timestamp นำหน้า) ต้องยังรู้ว่ากล้องไหน
    assert camera_of(Path("20260818_101500_IN_20260812_101500_123.jpg")) == "IN"
    assert camera_of(Path("20260818_101500_OUT_20260812_101500_123.jpg")) == "OUT"

    # log ต่อท้ายได้จริง และเขียนไม่ได้ต้องไม่ throw
    with tempfile.TemporaryDirectory() as tmp:
        g["YOLO_LOG"] = Path(tmp) / "sub" / "Yolo_is_detect.txt"
        log_yolo_hit(Path("IN_a.jpg"), 1, 1, 0.75)
        log_yolo_hit(Path("OUT_b.jpg"), 0, 2, 0.5)
        lines = g["YOLO_LOG"].read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2 and lines[0].endswith(
            "IN_a.jpg cam=IN car=1 plate=1 conf=0.75"
        )

        blocker = Path(tmp) / "blocker"
        blocker.write_bytes(b"x")  # parent เป็นไฟล์ → mkdir พัง แต่ต้องไม่ throw
        g["YOLO_LOG"] = blocker / "Yolo_is_detect.txt"
        log_yolo_hit(Path("IN_c.jpg"), 1, 1, 0.9)

        # log ส่งไม่สำเร็จ: ไฟล์เดิม retry ซ้ำต้องได้บรรทัดเดียว ไม่ใช่บรรทัดต่อรอบ
        g["FAILED_LOG"] = Path(tmp) / "filed_to_send.txt"
        _failed_logged.clear()
        for _ in range(3):
            log_failed(Path("IN_x.jpg"), "1กข1234", "HTTP 500: boom")
        log_failed(Path("OUT_y.jpg"), "2ขค5678", "เน็ต/backend: ConnectionError")
        lines = g["FAILED_LOG"].read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, lines
        assert lines[0].endswith("IN_x.jpg cam=IN plate=1กข1234 — HTTP 500: boom"), (
            lines
        )
        _failed_logged.clear()

    # --lag: คิดหน่วงจาก timestamp นำหน้า - mtime และต้องแยกกล้องออกจากชื่อไฟล์ใน send/
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shot = datetime(2026, 8, 18, 10, 15, 0)
        f = (
            d / "20260818_101507_IN_20260818_101500_123.jpg"
        )  # ถ่าย 10:15:00 อ่านจบ 10:15:07
        f.write_bytes(b"x")
        os.utime(f, (shot.timestamp(), shot.timestamp()))
        (d / "ยังไม่ย้าย_IN_x.jpg").write_bytes(b"x")  # ไม่มี timestamp นำหน้า = ต้องข้าม
        assert lag(d) == [(shot, 7.0, "IN")], lag(d)

    # จัดก้อนตามกล้อง + ช่วงเวลา: IN/OUT สลับกันมาต้องไม่ปนก้อนกัน
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        frames = [  # (ชื่อ, เวลาถ่าย) — IN คันแรก 3 เฟรม, OUT 1 เฟรม, IN คันที่สองหลังเว้น 5 วิ
            ("IN_1.jpg", 100.0),
            ("OUT_1.jpg", 100.2),
            ("IN_2.jpg", 100.5),
            ("IN_3.jpg", 101.0),
            ("IN_4.jpg", 106.0),
        ]
        for name, t in frames:
            f = d / name
            f.write_bytes(b"x")
            os.utime(f, (t, t))

        groups = group_frames(images_in(d))
        assert [[p.name for p in gp] for gp in groups] == [
            ["IN_1.jpg", "IN_2.jpg", "IN_3.jpg"],
            ["OUT_1.jpg"],
            ["IN_4.jpg"],
        ], groups

        # ก้อนที่เฟรมล่าสุดยังสดอยู่ = ยังไม่ปิด (รถอาจยังผ่านไม่หมด) เว้นแต่เฟรมล้น MAX_GROUP
        assert group_closed(groups[0], 101.0 + GROUP_GAP + 1)
        assert not group_closed(groups[0], 101.0 + GROUP_GAP - 0.1)
        assert group_closed(groups[0] * MAX_GROUP, 101.0)

    # รถต่อคิวกันมาไม่มีช่องว่าง (LINGER ทำให้ถ่ายต่อเนื่อง) ต้องถูกตัดทุก MAX_GROUP เฟรม
    # ไม่ใช่รวมเป็นก้อนเดียวแล้วได้ป้ายคันสุดท้ายคันเดียว
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for i in range(MAX_GROUP * 2 + 3):  # เฟรมห่างกัน 0.5 วิ ต่อเนื่องยาว
            f = d / f"IN_{i:03d}.jpg"
            f.write_bytes(b"x")
            os.utime(f, (100.0 + i * 0.5, 100.0 + i * 0.5))
        sizes = [len(gp) for gp in group_frames(images_in(d))]
        assert sizes == [MAX_GROUP, MAX_GROUP, 3], sizes

    # prune เก็บตัวใหม่สุด ทิ้งตัวเก่าสุด
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for i in range(5):
            f = d / f"{i}.jpg"
            f.write_bytes(b"x")
            os.utime(f, (i, i))
        prune(d, 2)
        assert sorted(p.name for p in d.iterdir()) == ["3.jpg", "4.jpg"]

    # ก้อนที่ประมวลผลพังต้องหลุดออกจากคิว ไม่ใช่ค้างวนพังซ้ำไม่จบ
    with tempfile.TemporaryDirectory() as tmp:
        g = globals()
        g["UNSEND_DIR"], g["SENT_DIR"] = Path(tmp) / "unsend", Path(tmp) / "send"
        g["UNSEND_DIR"].mkdir()
        poison = g["UNSEND_DIR"] / "IN_bad.jpg"
        poison.write_bytes(b"not a jpeg")
        os.utime(poison, (100.0, 100.0))  # เก่าพอให้ก้อนปิดแล้ว

        def boom(_):
            raise ValueError("ไฟล์เสีย")

        def stop(_):
            raise KeyboardInterrupt  # ถึง sleep = รอบนี้ไม่มีอะไรค้างให้ทำแล้ว

        real_time = time  # ต้องเก็บไว้ก่อน เพราะเดี๋ยวชื่อ time จะชี้ไปที่ของปลอม
        g["send_group"] = boom
        g["time"] = SimpleNamespace(time=time.time, sleep=stop)
        try:
            main()
        except KeyboardInterrupt:
            pass
        finally:
            g["time"] = real_time
        assert not images_in(g["UNSEND_DIR"]), "ไฟล์พิษยังค้างในคิว = จะวนพังตลอดไป"
        assert len(images_in(g["SENT_DIR"])) == 1

    print("✅ selfcheck ผ่าน")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _selfcheck()
    elif "--lag" in sys.argv:
        rest = sys.argv[sys.argv.index("--lag") + 1 :]
        lag(Path(rest[0]) if rest else None)
    elif "--batch" in sys.argv:
        # ดึง --score ออกก่อน เพื่อให้ตำแหน่ง [โฟลเดอร์] [กล้อง] หลัง --batch เหมือนเดิม
        truth_path = None
        if "--score" in sys.argv:
            i = sys.argv.index("--score")
            truth_path = Path(sys.argv[i + 1])
            del sys.argv[i : i + 2]

        rest = sys.argv[sys.argv.index("--batch") + 1 :]
        batch(
            Path(rest[0]),
            rest[1].upper() if len(rest) > 1 else None,
            truth_path,
        )
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\nหยุดแล้ว")
