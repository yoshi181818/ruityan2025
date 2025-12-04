from picamera2 import Picamera2
import cv2
from logic import process_frame

# カメラ初期化
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={
        "size": (960, 720),   # ★少し解像度アップ（Pi5なら余裕）
        "format": "BGR888"    # OpenCVで扱いやすいBGR
    }
)
picam2.configure(config)
picam2.start()


def generate_stream():
    """
    Flaskの /video_feed から呼ばれるジェネレータ。
    BGR → AI処理 → 表示用RGB → JPEG → MJPEGストリームへ。
    """
    while True:
        # カメラから BGR で取得
        frame = picam2.capture_array("main")

        # 顔検出 & 枠描画など（BGRのまま処理）
        frame = process_frame(frame)

        # 表示用：BGR → RGB（※チャンネル入れ替えはこれだけ）
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # JPEGにエンコード
        ret, buffer = cv2.imencode(".jpg", display)
        if not ret:
            continue

        jpg = buffer.tobytes()

        # MJPEGとしてクライアントへ送信
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
