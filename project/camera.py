from picamera2 import Picamera2
import cv2
from logic import process_frame

# カメラ初期化
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()


def generate_stream():
    """
    Flaskの /video_feed から呼ばれるジェネレータ。
    BGR → AI処理 → 表示用RGB → GBR → JPEG → ストリーム
    """
    while True:
        # カメラからBGRで取得
        frame = picam2.capture_array("main")

        # 顔検出 & 描画（BGRで処理）
        frame = process_frame(frame)

        # ↓↓↓ ここが "スタンドアロン版" と同じになる大事な部分 ↓↓↓

        # (1) AI処理後の BGR → RGB に変換
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # JPEG化
        ret, buffer = cv2.imencode(".jpg", display)
        if not ret:
            continue
        jpg = buffer.tobytes()

        # ストリーム出力
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
