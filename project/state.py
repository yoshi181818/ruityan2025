import time
import threading
import random
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
from flask import send_file

FACE_PATH = "/tmp/attitude_face.jpg"


class State:
    def __init__(self):
        # 設定値
        self.MEASURE_DURATION = 60   # 計測秒数
        self.BUZZER_PIN = 18

        # 状態
        self.lock = threading.Lock()
        self.analyzing = False
        self.start_time = None
        self.remaining = self.MEASURE_DURATION
        self.ng_counts = [0, 0, 0]       # [左, 中央, 右]
        self.winner = -1                 # 0/1/2 or -1
        self.reason = ""

        # GPIO初期化
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.BUZZER_PIN, GPIO.OUT)
        GPIO.output(self.BUZZER_PIN, GPIO.LOW)

    def reset(self):
        with self.lock:
            self.analyzing = True
            self.remaining = self.MEASURE_DURATION
            self.ng_counts = [0, 0, 0]
            self.winner = -1
            self.reason = ""

    def update_logic(self, current, latest_faces, frame, area_width):
        """
        current: [0/1, 0/1, 0/1] 今フレームで顔がいたかどうか
        latest_faces: 各エリアの最新顔画像
        frame: BGRフレーム（ここに赤枠を描画）
        area_width: 1エリアの幅
        """
        with self.lock:
            if self.analyzing and self.start_time is not None:
                elapsed = time.time() - self.start_time
                self.remaining = max(0, self.MEASURE_DURATION - elapsed)

                # 計測中フェーズ
                if elapsed < self.MEASURE_DURATION:
                    for i in range(3):
                        if current[i] == 0:
                            self.ng_counts[i] += 1
                else:
                    # 結果フェーズ
                    if self.winner == -1:
                        max_ng = max(self.ng_counts)
                        if max_ng > 0:
                            candidates = [i for i, n in enumerate(self.ng_counts)
                                          if n == max_ng]
                            self.winner = random.choice(candidates)
                            self.reason = "WORST ATTITUDE"
                        else:
                            self.winner = random.choice([0, 1, 2])
                            self.reason = "RANDOM (ALL GOOD)"

                        # 顔アップ保存
                        face = None
                        if 0 <= self.winner <= 2:
                            face = latest_faces[self.winner]
                        if face is not None:
                            cv2.imwrite(FACE_PATH, face)

                        # ブザー鳴動
                        GPIO.output(self.BUZZER_PIN, GPIO.HIGH)
                        time.sleep(1.0)
                        GPIO.output(self.BUZZER_PIN, GPIO.LOW)

                    self.analyzing = False

            # 選ばれたエリアに赤枠
            if self.winner != -1:
                sx = self.winner * area_width
                h = frame.shape[0]
                cv2.rectangle(frame,
                              (sx + 5, 5),
                              (sx + area_width - 5, h - 5),
                              (0, 0, 255), 5)

    def to_dict(self):
        with self.lock:
            return {
                "analyzing": self.analyzing,
                "remaining_time": int(self.remaining),
                "ng_counts": self.ng_counts,
                "selected_person": self.winner,
                "reason": self.reason,
            }

    def get_face_image(self):
        # ファイルがなければダミー生成
        if not os.path.exists(FACE_PATH):
            blank = np.zeros((260, 260, 3), dtype=np.uint8)
            cv2.imwrite(FACE_PATH, blank)
        return send_file(FACE_PATH, mimetype="image/jpeg")

    def cleanup(self):
        GPIO.cleanup()


# グローバルな状態インスタンス
state = State()
