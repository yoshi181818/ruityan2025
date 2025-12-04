import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from flask import send_file

FACE_PATH = "/tmp/face.jpg"

class State:
    def __init__(self):
        self.analyzing = False
        self.start_time = None
        self.remaining = 60
        self.ng_counts = [0,0,0]
        self.winner = -1
        self.reason = ""

        GPIO.setmode(GPIO.BCM)
        self.BUZZER_PIN = 18
        GPIO.setup(self.BUZZER_PIN, GPIO.OUT)

    def reset(self):
        self.analyzing = True
        self.remaining = 60
        self.ng_counts = [0,0,0]
        self.winner = -1
        self.reason = ""

    def update_logic(self, current, latest_faces, frame, area_w):
        if self.analyzing:
            elapsed = time.time() - self.start_time
            self.remaining = max(0, 60 - elapsed)

            # NGカウント
            if elapsed < 60:
                for i in range(3):
                    if current[i] == 0:
                        self.ng_counts[i] += 1
            else:
                if self.winner == -1:
                    mx = max(self.ng_counts)
                    if mx > 0:
                        candidates = [i for i,n in enumerate(self.ng_counts) if n == mx]
                        self.winner = candidates[0]
                        self.reason = "WORST ATTITUDE"
                    else:
                        self.winner = 1
                        self.reason = "RANDOM"

                    # 顔保存
                    face = latest_faces[self.winner]
                    if face is not None:
                        cv2.imwrite(FACE_PATH, face)

                    # ブザー
                    GPIO.output(self.BUZZER_PIN, GPIO.HIGH)
                    time.sleep(1)
                    GPIO.output(self.BUZZER_PIN, GPIO.LOW)

                self.analyzing = False

        # 赤枠
        if self.winner != -1:
            sx = self.winner * area_w
            cv2.rectangle(frame,(sx+5,5),(sx+area_w-5,frame.shape[0]-5),(0,0,255),5)

    def get_face_image(self):
        try:
            return send_file(FACE_PATH, mimetype="image/jpeg")
        except:
            blank = np.zeros((200,200,3),dtype=np.uint8)
            cv2.imwrite(FACE_PATH, blank)
            return send_file(FACE_PATH, mimetype="image/jpeg")

    def to_dict(self):
        return {
            "analyzing": self.analyzing,
            "remaining_time": int(self.remaining),
            "ng_counts": self.ng_counts,
            "selected_person": self.winner,
            "reason": self.reason,
        }

    def cleanup(self):
        GPIO.cleanup()

state = State()
