"""อ่านตัวอักษรจาก crop ป้าย ด้วยโมเดลทางการ th_PP-OCRv5_mobile_rec

อินเทอร์เฟซคือ recognize(BGR) → (text, score) — เขียน engine ตัวอื่นให้หน้าตาเหมือนนี้
แล้วแก้บรรทัด import ใน ocr.py บรรทัดเดียวก็สลับได้

ติดตั้ง — paddleocr ต้อง >=3.1 เพราะรุ่นภาษาไทยเพิ่งมาใน 3.1 (requirements.txt ปักไว้ 3.3.1)

    pip install 'paddleocr>=3.1'
    python3 rec_paddle.py --check   # โหลดโมเดลจริง + ลองอ่าน 1 ครั้ง

บน Pi เคยเจอว่า paddlepaddle ต้อง 3.0.0 เป๊ะ ไม่งั้น segfault ตอนโหลดโมเดล ส่วน
requirements.txt (freeze จากเครื่อง dev x86) ปักไว้ 3.3.0 — ค่าคนละตัวโดยตั้งใจ
ลงบน Pi แล้วเจอ segfault ให้ลด paddlepaddle ลงมา 3.0.0 ก่อนเป็นอย่างแรก

โมเดลดาวน์โหลดครั้งแรกครั้งเดียว แล้ว cache ไว้ที่ ~/.paddlex/official_models/
รันครั้งแรกต้องมีเน็ต
"""

import sys

MODEL_NAME = "th_PP-OCRv5_mobile_rec"
ENABLE_MKLDNN = False  # ARM64 ไม่มี oneDNN อยู่แล้ว บน x86 ปิดไว้ให้ผลนิ่ง
CPU_THREADS = 1  # เกิน 1 = segfault บน Pi

_ocr = None


def _load():
    """สร้าง predictor ครั้งเดียว — เรียกครั้งแรกจะโหลดโมเดล (ช้า/อาจดาวน์โหลด)"""
    global _ocr
    if _ocr is None:
        import numpy as np
        from paddleocr import TextRecognition

        _ocr = TextRecognition(
            model_name=MODEL_NAME,
            enable_mkldnn=ENABLE_MKLDNN,
            cpu_threads=CPU_THREADS,
        )
        # warm-up: เรียกแรกจริงๆ ช้ากว่าปกติหลายเท่า กินไปตั้งแต่ตอนโหลด ไม่ใช่ตอนรถคันแรกผ่าน
        _ocr.predict(np.zeros((48, 320, 3), dtype=np.uint8))
    return _ocr


def recognize(img):
    """crop ป้าย (BGR) → (text, score) — อ่านไม่ออกคืน ("", 0.0)

    predict() คืน generator ในบางเวอร์ชัน list ในบางเวอร์ชัน — list() ครอบไว้ทั้งคู่
    ถ้าไม่ครอบ `if not res` จะเป็น False เสมอบน generator แล้วพังตอน index
    """
    res = list(_load().predict(img))
    if not res:
        return "", 0.0

    r = res[0]
    try:
        return r["rec_text"], float(r["rec_score"])
    except (KeyError, TypeError):
        raise RuntimeError(f"ผลลัพธ์ paddle ไม่มี rec_text/rec_score — ได้ {type(r)}: {r!r}")


def _selfcheck():
    """python3 rec_paddle.py --check — โหลดโมเดลจริงแล้วลองอ่าน ต้องรันบนเครื่องที่จะใช้งาน"""
    import numpy as np

    text, score = recognize(np.full((48, 320, 3), 255, dtype=np.uint8))  # ภาพขาวล้วน
    assert isinstance(text, str), f"text ต้องเป็น str ได้ {type(text)}"
    assert isinstance(score, float), f"score ต้องเป็น float ได้ {type(score)}"
    assert 0.0 <= score <= 1.0, f"score หลุดกรอบ 0-1: {score} — OCR_CONF จะกรองผิดหมด"
    print(f"ภาพขาวล้วนอ่านได้ {text!r} score={score:.3f}")

    # ป้ายจริงคนละขนาดกับ 48x320 ต้อง resize เองไม่ได้ ต้องปล่อยให้ paddle จัดการ
    text, score = recognize(np.full((70, 200, 3), 255, dtype=np.uint8))
    assert isinstance(text, str) and 0.0 <= score <= 1.0

    print("✅ selfcheck ผ่าน — โมเดลโหลดได้ อ่านได้ score อยู่ในกรอบ")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _selfcheck()
    else:
        print(__doc__)
