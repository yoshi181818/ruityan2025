import cv2
import time
import threading
from logic import process_frame

# ★ラズパイのIPアドレス
RPI_URL = "http://192.168.100.252:5000/video_feed" 

class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.stream.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

camera_stream = VideoStream(RPI_URL)

def generate_stream():
    while True:
        success, frame = camera_stream.read()
        
        if not success or frame is None:
            print(f"再接続待機中: {RPI_URL}")
            time.sleep(1)
            continue

        # 1. リサイズ
        frame = cv2.resize(frame, (640, 480))

        # 2. ★色調整（ここを復活！）★
        # ラズパイからRGBで届いているため、BGRに変換して正常な色に戻します
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 3. AI処理
        frame = process_frame(frame)

        # 4. エンコード
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        jpg = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )