import cv2
import mediapipe as mp
from state import state

# Mediapipe 顔検出モデル
mp_face = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.6  # ★少しだけ厳しめ（精度寄り）
)

# 各エリアの最新顔画像（左・中央・右）
latest_faces = [None, None, None]


def preprocess_for_ai(frame_bgr):
    """
    顔検出しやすいように明るさとコントラストを少し整える前処理。
    入力: BGR画像
    出力: BGR画像（AI用）
    """
    # BGR → YCrCb にして輝度(Y)だけを調整
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # ヒストグラム均一化（CLAHEでやり過ぎないように）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)

    ycrcb = cv2.merge((y, cr, cb))
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # 軽くノイズを落とす（3x3の弱いぼかし）
    out = cv2.GaussianBlur(out, (3, 3), 0)

    return out


def process_frame(frame):
    """
    BGRフレームを受け取り、顔検出・判定・描画を行って BGR で返す。
    camera.py から呼ばれる。
    """
    # ★ AI用の前処理（明るさ・コントラスト調整）
    frame = preprocess_for_ai(frame)

    h, w, _ = frame.shape
    area_width = w // 3

    # Mediapipe用に BGR → RGB へ
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(frame_rgb)

    faces = []
    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            fw = int(box.width * w)
            fh = int(box.height * h)
            cx = x + fw // 2
            faces.append((cx, (x, y, fw, fh)))

    # 左から右へソート
    faces.sort(key=lambda f: f[0])

    # 今フレームで顔がいたかどうか
    current = [0, 0, 0]

    # 顔検出 & 最新顔保存 & 緑枠描画
    for cx, (x, y, fw, fh) in faces:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + fw)
        y1 = min(h, y + fh)

        face_crop = frame[y0:y1, x0:x1].copy()

        if cx < area_width:
            current[0] = 1
            latest_faces[0] = face_crop
        elif cx < area_width * 2:
            current[1] = 1
            latest_faces[1] = face_crop
        else:
            current[2] = 1
            latest_faces[2] = face_crop

        # 顔枠（緑）
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # エリアの区切り線を描画
    for i in range(1, 3):
        cv2.line(
            frame,
            (area_width * i, 0),
            (area_width * i, h),
            (200, 200, 200),
            1
        )

    # NGカウント・勝者決定・赤枠描画など（state側で処理）
    state.update_logic(current, latest_faces, frame, area_width)

    return frame  # BGRのまま camera.py に返す
