"""ฝั่งถ่ายภาพ — เปิด RTSP ค้างไว้ เจอความเปลี่ยนแปลงค่อยเซฟลง unsend/

รันคู่กับ Run_v2.py ซึ่งเป็นฝั่งอ่านป้าย+ส่ง backend (คนละโปรเซส):

    python3 Run_v2_capture.py IN       # ฝั่งถ่าย กล้องทางเข้า
    python3 Run_v2_capture.py OUT      # ฝั่งถ่าย กล้องทางออก (คนละโปรเซส)
    python3 Run_v2.py                  # ฝั่งอ่าน+ส่ง — ตัวเดียวพอ รับของจากทั้ง 2 กล้อง
    python3 Run_v2_capture.py --check  # เช็คตรรกะ motion ไม่ต้องมีกล้อง
    python3 Run_v2_capture.py IN --tune  # ดูค่า motion จริงของกล้องนี้ ไม่เซฟรูป

ponytail: ไม่ import Run_v2 มาใช้ค่าคงที่ร่วมกัน เพราะ Run_v2 โหลด YOLO+Paddle
ตอน import โปรเซสถ่ายภาพจะกิน RAM เป็น GB ฟรีๆ ก๊อป 2 ค่ามาดีกว่า
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


def load_dotenv(path=Path(__file__).with_name(".env")):
    """อ่าน KEY=VALUE จาก .env ข้างไฟล์นี้ใส่ os.environ — ค่าที่ตั้งใน shell อยู่แล้วชนะเสมอ

    ponytail: parser ง่ายๆ ไม่รองรับ multiline/export/${VAR} — ต้องการค่อยเปลี่ยนไปใช้ python-dotenv
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key or key.startswith("#"):
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]  # ลอกเฉพาะ quote ที่ครอบคู่กัน รหัสที่ลงท้ายด้วย ' ต้องไม่โดนตัด
        os.environ.setdefault(key, val)


load_dotenv()

# ===== ตั้งค่า =====
UNSEND_DIR = Path("/home/pi/pi-edge_cloud/unsend")  # ต้องตรงกับใน Run_v2.py
MAX_UNSEND = 200  # ฝั่งอ่านตามไม่ทัน/เน็ตล่ม เกินนี้ทิ้งรูปเก่าสุด กัน SD card เต็ม

SMALL_W = 320  # ย่อก่อนเทียบ เร็วขึ้นและกัน noise เซนเซอร์ไปในตัว
PIXEL_DELTA = 25  # ความสว่างต่างเกินนี้ถือว่าพิกเซลนั้นเปลี่ยน (0-255)
SAVE_INTERVAL = 0.5  # เซฟถี่สุดกี่วินาที/รูป — ฝั่งอ่านใช้ ~0.56 วิ/รูป อย่าป้อนเร็วกว่านั้นมาก
LINGER = 3.0  # เห็นการเคลื่อนไหวแล้วถ่ายต่ออีกกี่วินาที แม้ภาพจะนิ่ง (จับรถที่จอดแล้ว) — 0 = ปิด
# ค่านี้เป็น default ตั้งแยกต่อกล้องได้ในตาราง CAMERAS ข้างล่าง (กล้องหน้าไม้กั้นต้องนานกว่า)
RECONNECT_DELAY = 3  # สตรีมหลุดแล้วรอกี่วินาทีก่อนต่อใหม่

# กล้องแต่ละตัว — ชื่อ (key) คือ pk ที่ติดไปกับรูปและส่งขึ้น backend เป็นฟิลด์ camera
# roi = โซนที่สนใจ สัดส่วน 0-1 (x1, y1, x2, y2), None = ทั้งภาพ
# ตั้ง roi เมื่อในเฟรมมีต้นไม้/ป้ายโบกที่ทำให้ทริกเกอร์ทำงานตลอด — ตั้งแยกได้ต่อกล้อง
# motion = สัดส่วนพิกเซลที่เปลี่ยนระหว่าง 2 เฟรมติดกัน ที่ถือว่า "มีอะไรเคลื่อนไหว"
#   ต่ำ = ไวขึ้น จับรถตั้งแต่ยังไกล แต่ใบไม้ไหว/เงาเมฆก็ทริกเกอร์ด้วย
#   สูง = ถ่ายเฉพาะตอนรถใกล้แล้ว มุมป้ายเบ้กว่า อ่านยากกว่า
#   คนละกล้องคนละมุม/ระยะ ต้องจูนแยกกัน ห้ามใช้ค่าเดียวกันเพราะ "มันเคยใช้ได้"
# linger = ถ่ายต่อกี่วินาทีหลังภาพนิ่ง (ไม่ใส่ = ใช้ LINGER ข้างบน)
#   กล้องที่รถวิ่งผ่านไม่หยุด 3 วิพอ กล้องหน้าไม้กั้นที่รถจอดรอต้องนานกว่าเวลาที่รถจอด
#   ไม่งั้นจะหยุดถ่ายตอนที่ป้ายนิ่งและคมที่สุดพอดี
# channel = ช่อง RTSP ของ NVR/กล้อง (101 = ช่อง 1 สตรีมหลัก, 102 = ช่อง 2 สตรีมหลัก)
CAMERAS = {
    "IN": {
        "ip": "192.168.1.64",
        "channel": 101,
        "roi": None,
        "motion": 0.05,
        # หน้าไม้กั้น: รถคลานเข้ามา จอดรอไม้กั้นเปิด แล้วค่อยขยับ ช่วงที่จอดนิ่งคือช่วงที่ป้าย
        # คมที่สุด แต่ delta ระหว่างเฟรมเป็น 0 ถ้า linger สั้นจะหยุดถ่ายตอนนั้นพอดี
        # 15 วิ = เผื่อเวลารอไม้กั้นตามปกติ จอดรอนานกว่านี้ค่อยดันขึ้น
        "linger": 15.0,
    },
    "OUT": {
        "ip": "192.168.2.64",
        "channel": 101,
        "roi": None,
        "motion": 0.05,
        "linger": 3.0,  # รถขาออกไม่ต้องหยุดรอ วิ่งผ่านตลอด 3 วิพอ
    },
}

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

CAM = next((a.upper() for a in sys.argv[1:] if not a.startswith("-")), "IN")
if CAM not in CAMERAS:
    sys.exit(f"ไม่รู้จักกล้อง {CAM!r} — มีให้เลือก: {', '.join(CAMERAS)}")
_cfg = CAMERAS[CAM]
CAMERA_IP = _cfg["ip"]
ROI = _cfg["roi"]
# จูนผ่าน env ได้ ไม่ต้องแก้ไฟล์: CAM_MOTION_IN=0.01 CAM_LINGER_IN=15 python3 Run_v2_capture.py IN --tune
MOTION_RATIO = float(os.environ.get(f"CAM_MOTION_{CAM}") or _cfg["motion"])
LINGER = float(os.environ.get(f"CAM_LINGER_{CAM}") or _cfg.get("linger") or LINGER)
# รหัสกล้องอยู่ใน .env เท่านั้น (ดู .env.example) — ไม่มี fallback ในโค้ด กันรหัสหลุดขึ้น git อีก
# CAM_PASSWORD เฉยๆ ครอบทุกกล้อง — ตั้งค้างไว้จากตอนกล้องเดียว กล้องที่ 2 จะได้รหัสผิดเงียบๆ
CAM_PASSWORD = os.environ.get(f"CAM_PASSWORD_{CAM}", "")
if not CAM_PASSWORD and "--check" not in sys.argv:
    sys.exit(f"ไม่มี CAM_PASSWORD_{CAM} — ก๊อป .env.example เป็น .env แล้วใส่รหัสกล้อง")
CHANNEL = int(os.environ.get(f"CAM_CHANNEL_{CAM}") or _cfg["channel"])
RTSP_URL = f"rtsp://admin:{CAM_PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/{CHANNEL}"
# ==================


def prep(frame):
    """ย่อ + เทา + เบลอ ให้พร้อมเทียบ — คืนภาพเล็กที่ใช้ตรวจความเปลี่ยนแปลง"""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (SMALL_W, max(1, int(h * SMALL_W / w))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if ROI:
        gh, gw = gray.shape
        x1, y1, x2, y2 = ROI
        gray = gray[int(y1 * gh) : int(y2 * gh), int(x1 * gw) : int(x2 * gw)]

    return cv2.GaussianBlur(gray, (5, 5), 0)


def motion_ratio(prev, cur):
    """สัดส่วนพิกเซลที่เปลี่ยนไประหว่าง 2 เฟรม (0.0-1.0)"""
    diff = cv2.absdiff(prev, cur)
    _, mask = cv2.threshold(diff, PIXEL_DELTA, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(mask) / mask.size


def should_save(now, last_motion, last_save):
    """ยังอยู่ในช่วง LINGER หลังการเคลื่อนไหวล่าสุด และเว้นระยะจากรูปก่อนพอแล้ว"""
    return now - last_motion <= LINGER and now - last_save >= SAVE_INTERVAL


def open_stream():
    """เปิด RTSP — คืน None ถ้าเปิดไม่ได้ ให้ผู้เรียกลองใหม่เอง"""
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None

    # กันเฟรมค้างสะสม: ตอนเราไปเซฟไฟล์อยู่ สตรีมยังไหลเข้าบัฟเฟอร์เรื่อยๆ
    # ถ้าไม่จำกัดจะได้ประมวลผลภาพย้อนหลังแล้วหน่วงสะสมขึ้นเรื่อยๆ
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def make_room(d: Path, keep: int):
    """คิวเต็ม = ทิ้งรูปเก่าสุด ไม่ใช่ข้ามการเซฟรูปใหม่

    ponytail: ของเดิมพอคิวเต็มแล้วหยุดเซฟ = รถที่กำลังวิ่งเข้ามาตอนนี้หายเลย ทั้งที่รูปเมื่อ
    5 นาทีที่แล้วยังกองอยู่ ระบบหน้าด่านต้องเลือกของสด ไฟล์อาจโดนฝั่งอ่านย้ายไปพร้อมกัน
    (คนละโปรเซส) เลยต้อง missing_ok
    """
    files = sorted(
        (p for p in d.glob("*.jpg") if p.is_file()), key=lambda p: p.stat().st_mtime
    )
    for old in files[: len(files) - keep + 1]:
        old.unlink(missing_ok=True)
        print(f"🗑️  คิวเต็ม ทิ้ง {old.name} (ฝั่งอ่านตามไม่ทันหรือเปล่า)")


def save(frame, ratio):
    """เซฟลง unsend/ — ใส่มิลลิวินาทีกันชื่อชนตอนเซฟถี่ๆ"""
    make_room(UNSEND_DIR, MAX_UNSEND)

    # prefix ชื่อกล้อง = pk ที่ฝั่งอ่านใช้บอกว่ารูปนี้มาจากทางเข้าหรือทางออก
    name = f"{CAM}_" + f"{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3] + ".jpg"
    # เขียนเป็น .tmp ก่อนแล้ว rename — ฝั่งอ่านมองไม่เห็น .tmp เลยไม่มีทางคว้าไฟล์ที่เขียนค้างอยู่
    # imencode ไม่ใช่ imwrite เพราะ imwrite เลือก codec จากนามสกุล และไม่รู้จัก .tmp
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        print("⚠️  encode jpg ไม่สำเร็จ ข้ามเฟรมนี้")
        return

    tmp = UNSEND_DIR / (name + ".tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.rename(UNSEND_DIR / name)
    print(f"📸 {name} (เปลี่ยน {ratio * 100:.1f}%)")


def main():
    UNSEND_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"เฝ้ากล้อง {CAM} ({CAMERA_IP}) — เซฟลง {UNSEND_DIR} "
        f"เมื่อภาพเปลี่ยนเกิน {MOTION_RATIO * 100:.1f}%"
    )
    print("Ctrl+C เพื่อหยุด")

    cap = None
    prev = None
    last_save = 0.0
    last_motion = -LINGER - 1  # ยังไม่เคยเห็นอะไรเคลื่อนไหว

    while True:
        if cap is None:
            cap = open_stream()
            if cap is None:
                print(f"❌ ต่อกล้องไม่ได้ — ลองใหม่ใน {RECONNECT_DELAY} วิ")
                time.sleep(RECONNECT_DELAY)
                continue
            print("✅ ต่อกล้องได้แล้ว")
            prev = None  # เฟรมก่อนหน้าใช้ไม่ได้แล้ว เริ่มนับใหม่

        ok, frame = cap.read()
        if not ok:
            print("⚠️  อ่านเฟรมไม่ได้ — ต่อใหม่")
            cap.release()
            cap = None
            continue

        cur = prep(frame)
        if prev is None:  # เฟรมแรกไม่มีอะไรให้เทียบ
            prev = cur
            continue

        ratio = motion_ratio(prev, cur)
        prev = cur

        # ponytail: เทียบเฟรมติดกัน = จับ "การเคลื่อนไหว" ไม่ใช่ "การมีอยู่" รถจอดนิ่งแล้ว
        # จะหยุดทริกเกอร์ LINGER เลยลากการถ่ายต่ออีก 3 วิ ให้ได้เฟรมตอนรถหยุด (คมสุด)
        # ถ้าหน้างานเป็นด่านที่รถจอดรอนานกว่านั้น ค่อยเปลี่ยนไปใช้ background subtraction
        now = time.time()
        if ratio >= MOTION_RATIO:
            last_motion = now
        if should_save(now, last_motion, last_save):
            save(frame, ratio)
            last_save = now


def tune():
    """ดูค่า motion จริงของกล้องตัวนี้ ไม่เซฟรูป — ใช้ตั้ง MOTION_RATIO ต่อกล้อง

    เปิดทิ้งไว้แล้วดู 2 อย่าง: ตอนไม่มีรถค่าขึ้นไปแค่ไหน (= พื้นเสียง ต้องอยู่ใต้เส้น)
    กับตอนรถวิ่งผ่านค่าพุ่งเท่าไหร่ (= ต้องอยู่เหนือเส้น) แล้วตั้งเส้นไว้ตรงกลาง
    ถ้าพื้นเสียงสูงตลอดเพราะมีต้นไม้/ถนนหลังฉาก ให้ตั้ง roi ตัดโซนนั้นทิ้งแทนการดันเส้นขึ้น
    """
    cap = open_stream()
    if cap is None:
        sys.exit(f"❌ ต่อกล้อง {CAM} ({CAMERA_IP}) ไม่ได้")
    print(f"จูน {CAM} ({CAMERA_IP}) — เส้นปัจจุบัน {MOTION_RATIO * 100:.1f}% | Ctrl+C หยุด")

    prev, peak = None, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  อ่านเฟรมไม่ได้")
            break
        cur = prep(frame)
        if prev is None:
            prev = cur
            continue
        ratio = motion_ratio(prev, cur)
        prev = cur
        peak = max(peak, ratio)
        # แถบละ 1% เส้นคือ MOTION_RATIO — เห็นด้วยตาว่าเฟรมไหนข้ามเส้น
        bar = "█" * int(ratio * 100)
        mark = "📸 ถึงเส้น" if ratio >= MOTION_RATIO else ""
        print(f"{ratio * 100:5.2f}% (สูงสุด {peak * 100:5.2f}%) {bar} {mark}")


def _selfcheck():
    """python3 Run_v2_capture.py --check — เช็คตรรกะ motion ไม่ต้องมีกล้อง"""
    import numpy as np

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    same = prep(blank)
    assert motion_ratio(same, same) == 0.0, "ภาพเดิมต้องไม่มีการเปลี่ยนแปลง"

    noisy = blank.copy()  # จุดรบกวนเล็กๆ ต้องไม่ปลุก YOLO
    noisy[0:5, 0:5] = 255
    assert motion_ratio(same, prep(noisy)) < MOTION_RATIO, "จุดเล็กๆ ต้องไม่ทริกเกอร์"

    car = blank.copy()  # วัตถุขนาดรถกินพื้นที่ราว 10% ต้องทริกเกอร์
    car[150:350, 200:450] = 255
    assert motion_ratio(same, prep(car)) >= MOTION_RATIO, "วัตถุใหญ่ต้องทริกเกอร์"

    # คิวเต็มต้องทิ้งรูปเก่าสุด แล้วเหลือที่ว่างให้รูปใหม่ 1 ใบเสมอ
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for i in range(5):
            f = d / f"{i}.jpg"
            f.write_bytes(b"x")
            os.utime(f, (i, i))
        make_room(d, 3)
        assert sorted(p.name for p in d.iterdir()) == ["3.jpg", "4.jpg"], (
            "ต้องเหลือของใหม่"
        )
        make_room(d, 9)  # ยังไม่เต็ม ห้ามลบอะไร
        assert len(list(d.iterdir())) == 2

        # .env: ข้าม comment, = ในรหัสต้องอยู่ครบ, ลอก quote คู่, ค่าใน shell ต้องชนะไฟล์
        env = d / ".env"
        env.write_text("# c\n_T1=a=b\n_T2='x#y'\n_T3=file\n_T4=ab'\n", encoding="utf-8")
        os.environ["_T3"] = "shell"
        load_dotenv(env)
        got = [os.environ.pop(k) for k in ("_T1", "_T2", "_T3", "_T4")]
        assert got == ["a=b", "x#y", "shell", "ab'"], got

    t = 1000.0  # เห็นการเคลื่อนไหวล่าสุดที่ t
    assert should_save(t + 1.0, t, t - 5), "รถจอดนิ่ง 1 วิ ยังต้องถ่ายอยู่"
    assert not should_save(t + 1.0, t, t + 0.9), "เพิ่งถ่ายไป 0.1 วิ ต้องยังไม่ถ่ายซ้ำ"
    assert not should_save(t + LINGER + 1, t, t - 5), "เลย LINGER แล้วต้องหยุดถ่าย"

    print("✅ selfcheck ผ่าน")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _selfcheck()
    elif "--tune" in sys.argv:
        try:
            tune()
        except KeyboardInterrupt:
            print("\nหยุดแล้ว")
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\nหยุดแล้ว")
