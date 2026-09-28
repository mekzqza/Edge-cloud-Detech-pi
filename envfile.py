"""โหลด .env ใส่ os.environ — ใช้ร่วมกันระหว่าง Run_v2.py กับ Run_v2_capture.py

แยกไฟล์เพราะทั้งสองฝั่ง import กันเองไม่ได้: Run_v2 โหลด YOLO+Paddle, Run_v2_capture เลือกกล้อง
และ sys.exit ตั้งแต่ตอน import
"""

import os
from pathlib import Path


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
