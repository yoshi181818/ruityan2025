import time
import cv2
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

# ---------------------------------------------------------
# カメラ初期化
# ---------------------------------------------------------
# ネットワーク負荷を下げるため、解像度は640x480程度に抑えるのがコツです
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),
        "format": "BGR888"
    }
)
picam2.configure(config)
picam2.start()

def generate_stream():
    """
    カメラから画像を取得し、JPEGに圧縮して配信するジェネレータ
    """
    while True:
        # 1. 画像取得 (BGR形式)
        frame = picam2.capture_array("main")
        
        # ※もし色が青っぽい/赤っぽい場合は、以下のコメントを外して色変換を試してください
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2. JPEG圧縮 (品質80で通信量削減)
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
            
        jpg = buffer.tobytes()

        # 3. マルチパートストリームとして送信
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # 0.0.0.0 で全ネットワークに向けて公開
    print("★ラズパイカメラ配信中... http://<ラズパイのIP>:5000/video_feed")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)