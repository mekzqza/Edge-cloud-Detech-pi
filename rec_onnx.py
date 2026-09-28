"""อ่านตัวอักษรจาก crop ป้าย ด้วยโมเดล fine-tune ที่แปลงเป็น ONNX

อินเทอร์เฟซเหมือน rec_paddle เป๊ะ: recognize(BGR) → (text, score)
สลับกลับไป paddle ได้ด้วยการแก้บรรทัด `import rec_onnx as rec` ใน ocr.py บรรทัดเดียว

ทำไมต้อง ONNX แทนที่จะเรียก paddle ตรงๆ
---------------------------------------
paddlepaddle บน Pi เป็น 3.0.0 (ล็อกไว้เพราะเป็นเวอร์ชันที่มี wheel ให้ ARM64 และเป็นเวอร์ชัน
ที่ไม่ segfault — ดู rec_paddle.py) แต่ model_ocr/inference_final_1 ถูก export ด้วย
paddlepaddle 3.3.0 ซึ่งเขียน attribute `strides` ใน PIR เป็น int64 ส่วน 3.0.0 คาด int32
พอโหลดเลยตาย ValueError: (InvalidArgument) Type of attribute: strides is not right

onnxruntime อ่าน .onnx โดยไม่สน PIR หรือเวอร์ชัน paddle เลย และ Pi ก็มี onnxruntime
ใช้รัน YOLO อยู่แล้ว (detect.py / ocr.py) — ฝั่ง OCR เลยเลิกพึ่ง paddlepaddle ทั้งก้อน

แปลง .onnx ใหม่เมื่อเทรนโมเดลรอบหน้า — รันบนเครื่องที่ paddle อ่าน inference.json ได้
(ต้อง >= เวอร์ชันที่ export มา ไม่ใช่บน Pi):

    pip install paddle2onnx
    python -c "import paddle2onnx; paddle2onnx.export( \\
        'model/th_plate_rec/inference.json', \\
        'model/th_plate_rec/inference.pdiparams', \\
        'model/th_plate_rec/rec.onnx', opset_version=14)"

ตรวจว่าแปลงมาไม่เพี้ยน — ป้อน tensor เดียวกันเข้าทั้ง onnx และ paddle แล้วเทียบ logits
รอบที่แปลงไฟล์นี้ได้ max abs diff = 2.1e-06 และ argmax ตรงกันทุกตำแหน่ง

    python3 rec_onnx.py --check
"""

import sys
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).parent / "model" / "th_plate_rec" / "rec.onnx"
CONFIG = Path(__file__).parent / "model" / "th_plate_rec" / "inference.yml"

# ตรึงตามที่โมเดลถูกเทรนมา — inference.yml: RecResizeImg.image_shape = [3, 48, 320]
# input ของ .onnx เป็น [N, 3, 48, dynamic] จะป้อนกว้างกว่า 320 ก็ไม่ error แต่โมเดล
# ไม่เคยเห็นภาพแบบนั้นตอนเทรน ความแม่นจะตกเงียบๆ — อยากลองต้องวัดผลก่อน
IMG_H, IMG_W = 48, 320

# ponytail: 1 เธรดเหมือน rec_paddle.CPU_THREADS — Pi รัน YOLO อยู่ด้วย แย่งคอร์กันแล้ว
# ทั้งคู่ช้าลง ถ้าวัดแล้วพบว่า OCR เป็นคอขวดจริงค่อยขยับขึ้น 2
ORT_THREADS = 1

_sess = None
_charset = None


def _load():
    """สร้าง session + อ่าน charset ครั้งเดียว — เรียกครั้งแรกจะโหลดโมเดล"""
    global _sess, _charset
    if _sess is None:
        import onnxruntime as ort
        import yaml

        if not MODEL.exists():
            raise FileNotFoundError(f"ไม่พบโมเดล OCR ที่ {MODEL} — ดูวิธีแปลงใน docstring")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ORT_THREADS
        opts.inter_op_num_threads = ORT_THREADS
        _sess = ort.InferenceSession(
            str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        # charset ต้องเรียงแบบเดียวกับ CTCLabelDecode ของ paddle เป๊ะ: blank อยู่ index 0
        # ตามด้วย character_dict แล้วปิดท้ายด้วย space — เรียงผิดแม้ตัวเดียว = อ่านออกมา
        # เป็นตัวอักษรอื่นทั้งใบโดยไม่มี error ให้เห็น
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        _charset = ["<blank>"] + list(cfg["PostProcess"]["character_dict"]) + [" "]

        n_classes = _sess.get_outputs()[0].shape[-1]
        if len(_charset) != n_classes:
            raise RuntimeError(
                f"charset {len(_charset)} ตัว แต่โมเดลมี {n_classes} class — "
                f"inference.yml กับ rec.onnx มาจากคนละรอบ export"
            )

        _sess.run(None, {"x": np.zeros((1, 3, IMG_H, IMG_W), dtype=np.float32)})  # warm-up
    return _sess, _charset


def _preprocess(img):
    """BGR ขนาดใดก็ได้ → [1, 3, 48, 320] — ตรงตาม resize_norm_img ของ PaddleOCR

    ย่อให้สูง 48 โดยคงอัตราส่วน ถ้ากว้างเกิน 320 ตัดที่ 320 (บีบภาพ) ที่เหลือ pad ขวา
    ด้วย 0 ซึ่งหลัง normalize แล้วเท่ากับเทากลาง ไม่ใช่ดำ — ตรงกับที่โมเดลเห็นตอนเทรน
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("ภาพว่าง")

    resized_w = min(IMG_W, max(1, int(np.ceil(IMG_H * w / h))))
    resized = cv2.resize(img, (resized_w, IMG_H)).astype(np.float32)
    resized = resized.transpose(2, 0, 1) / 255.0
    resized = (resized - 0.5) / 0.5

    out = np.zeros((3, IMG_H, IMG_W), dtype=np.float32)
    out[:, :, :resized_w] = resized
    return out[None]


def _ctc_collapse(probs):
    """[T, C] softmax → (ข้อความ, prob ของตัวที่เก็บไว้) — CTC greedy แบบเดียวกับ paddle

    ยุบตัวซ้ำที่ติดกันก่อน แล้วค่อยทิ้ง blank (ลำดับสลับกันไม่ได้ ไม่งั้น "กก" จะเหลือ "ก")
    """
    idx = probs.argmax(-1)
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = idx[1:] != idx[:-1]
    keep &= idx != 0

    _, charset = _load()
    text = "".join(charset[i] for i in idx[keep])
    return text, probs[keep, idx[keep]]


def _score(char_probs):
    """prob ของตัวอักษรที่อ่านได้ (array 1 มิติ) → ค่า confidence ตัวเดียว 0.0-1.0

    ใช้ mean ให้ตรงกับ CTCLabelDecode ของ paddle — ไม่ใช่เพราะ mean ดีที่สุด แต่เพราะ
    conf_threshold กับ PROVINCE_CONF (0.15) ใน ocr.py ถูกจูนมากับคะแนนที่ paddle ให้
    เปลี่ยนสูตรตรงนี้ = ตัวเลขพวกนั้นผิดความหมายทันทีโดยไม่มีอะไรฟ้อง และตอนนี้ยังไม่มี
    ชุดภาพไว้คาลิเบรตใหม่

    ponytail: ถ้าเจอป้ายที่ผิดตัวเดียวหลุด threshold บ่อย ให้ลอง min เป็นอันดับแรก —
    งานนี้ผิดตัวเดียวคือผิดทั้งป้าย ซึ่ง mean กลบให้ได้ (6 ตัวมั่นใจ + 1 ตัวเดา 0.3
    เฉลี่ยแล้วยังดูดี) แต่ต้องวัดกับภาพจริงและลด PROVINCE_CONF ตามก่อนเปลี่ยน
    """
    return float(char_probs.mean())


def recognize(img):
    """crop ป้าย (BGR) → (text, score) — อ่านไม่ออกคืน ("", 0.0)"""
    sess, _ = _load()
    probs = sess.run(None, {"x": _preprocess(img)})[0][0]  # [T, C] softmax แล้วจากโมเดล
    text, char_probs = _ctc_collapse(probs)
    if not text:
        return "", 0.0
    return text, float(_score(char_probs))


def _selfcheck():
    """python3 rec_onnx.py --check — โหลดโมเดลจริงแล้วลองอ่าน ต้องรันบนเครื่องที่จะใช้งาน"""
    sess, charset = _load()
    print(f"โมเดล {MODEL.name} | {len(charset)} class | input {sess.get_inputs()[0].shape}")

    text, score = recognize(np.full((48, 320, 3), 255, dtype=np.uint8))  # ภาพขาวล้วน
    assert isinstance(text, str), f"text ต้องเป็น str ได้ {type(text)}"
    assert isinstance(score, float), f"score ต้องเป็น float ได้ {type(score)}"
    assert 0.0 <= score <= 1.0, f"score หลุดกรอบ 0-1: {score} — OCR_CONF จะกรองผิดหมด"
    print(f"ภาพขาวล้วนอ่านได้ {text!r} score={score:.3f}")

    # ป้ายจริงคนละขนาดกับ 48x320 ต้อง resize เองไม่ได้ ต้องปล่อยให้ _preprocess จัดการ
    text, score = recognize(np.full((70, 200, 3), 255, dtype=np.uint8))
    assert isinstance(text, str) and 0.0 <= score <= 1.0

    # ภาพที่กว้างกว่า 320/48 ต้องถูกบีบ ไม่ใช่ทำให้ tensor ผิดขนาดแล้ว onnx โยน error
    assert _preprocess(np.zeros((40, 4000, 3), dtype=np.uint8)).shape == (1, 3, 48, 320)

    print("✅ selfcheck ผ่าน — โมเดลโหลดได้ อ่านได้ score อยู่ในกรอบ")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _selfcheck()
    else:
        print(__doc__)
