from picamera2 import Picamera2
import cv2
from logic import process_frame

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640,480), "format":"BGR888"}
)
picam2.configure(config)
picam2.start()

def generate_stream():
    while True:
        frame = picam2.capture_array("main")
        processed = process_frame(frame)

        # GBRに変換
        disp = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        disp = disp[:, :, [1,2,0]]

        ret, buffer = cv2.imencode(".jpg", disp)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")
