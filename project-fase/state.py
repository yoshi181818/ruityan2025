import time
import threading
import random
import os
import cv2
import numpy as np
from flask import send_file

FACE_PATH = "attitude_face.jpg"

class State:
    def __init__(self):
        self.MEASURE_DURATION = 30
        self.lock = threading.Lock()
        self.analyzing = False
        self.start_time = None
        self.remaining = self.MEASURE_DURATION
        self.person_scores = {} # {"Name": Score}
        self.person_faces = {}
        self.display_ng_counts = [0, 0, 0]
        self.winner = -1
        self.reason = ""

    def reset(self):
        with self.lock:
            self.analyzing = True
            self.start_time = time.time()
            self.remaining = self.MEASURE_DURATION
            self.person_scores = {}
            self.person_faces = {}
            self.display_ng_counts = [0, 0, 0]
            self.winner = -1
            self.reason = ""

    def update_logic_recognition(self, detected_people, frame):
        with self.lock:
            if self.analyzing and self.start_time:
                elapsed = time.time() - self.start_time
                self.remaining = max(0, self.MEASURE_DURATION - elapsed)

                if elapsed < self.MEASURE_DURATION:
                    # 計測中
                    for name, (x, y, w, h), penalty in detected_people:
                        # Unknownを除外したい場合はここで if name == "Unknown": continue
                        if name not in self.person_scores:
                            self.person_scores[name] = 0.0
                        
                        if penalty > 0:
                            self.person_scores[name] += penalty

                        # 顔保存
                        if w > 0 and h > 0:
                            self.person_faces[name] = frame[y:y+h, x:x+w].copy()

                    # ワースト3
                    sorted_p = sorted(self.person_scores.items(), key=lambda x: x[1], reverse=True)
                    self.display_ng_counts = [0, 0, 0]
                    for i in range(min(3, len(sorted_p))):
                        self.display_ng_counts[i] = int(sorted_p[i][1] * 10)
                else:
                    # 結果発表
                    if self.winner == -1:
                        if not self.person_scores:
                            self.reason = "NOBODY"
                            self.winner = -1
                        else:
                            worst_name, score = max(self.person_scores.items(), key=lambda x: x[1])
                            if score > 0:
                                self.winner = 0
                                self.reason = f"WORST: {worst_name}"
                                if worst_name in self.person_faces:
                                    cv2.imwrite(FACE_PATH, self.person_faces[worst_name])
                            else:
                                self.winner = random.choice([0,1,2])
                                self.reason = "ALL GOOD"
                    self.analyzing = False

    def to_dict(self):
        with self.lock:
            return {
                "analyzing": self.analyzing,
                "remaining_time": int(self.remaining),
                "ng_counts": self.display_ng_counts, 
                "selected_person": self.winner,
                "reason": self.reason,
            }

    def get_face_image(self):
        if not os.path.exists(FACE_PATH):
            blank = np.zeros((260, 260, 3), dtype=np.uint8)
            cv2.imwrite(FACE_PATH, blank)
        return send_file(FACE_PATH, mimetype="image/jpeg")

    def cleanup(self): pass

state = State()
