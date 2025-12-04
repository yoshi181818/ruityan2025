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
    BGR → AI処理 → 表示用GBR → JPEG → MJPEGストリーム
    """
    while True:
        frame = picam2.capture_array("main")  # BGR

        # 顔検出 & 描画 & 判定
        frame = process_frame(frame)

        # 表示用：BGR → RGB → GBR に並び替え
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB
        display = display[:, :, [1, 2, 0]]                # RGB → GBR

        ret, buffer = cv2.imencode(".jpg", display)
        if not ret:
            continue
        jpg = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
