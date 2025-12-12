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
        self.MEASURE_DURATION = 60
        self.lock = threading.Lock()
        self.analyzing = False
        self.start_time = None
        self.remaining = self.MEASURE_DURATION
        
        # ★変更: IDごとのスコア管理 { track_id: ng_score }
        self.person_scores = {}
        # 顔画像の一時保存 { track_id: image }
        self.person_faces = {}
        
        # 画面表示用に変換した結果 [ワースト1位, 2位, 3位]
        self.display_ng_counts = [0, 0, 0]
        
        self.winner = -1
        self.reason = ""

    def reset(self):
        with self.lock:
            self.analyzing = True
            self.remaining = self.MEASURE_DURATION
            self.person_scores = {}
            self.person_faces = {}
            self.display_ng_counts = [0, 0, 0]
            self.winner = -1
            self.reason = ""

    def update_logic_tracking(self, detected_tracks, frame):
        """
        detected_tracks: [(track_id, cx, (x, y, w, h), penalty), ...]
        """
        with self.lock:
            # 1. タイム計測
            if self.analyzing and self.start_time is not None:
                elapsed = time.time() - self.start_time
                self.remaining = max(0, self.MEASURE_DURATION - elapsed)

                if elapsed < self.MEASURE_DURATION:
                    # ------------------------------------
                    # 計測中: IDごとにスコア加算
                    # ------------------------------------
                    current_ids = set()
                    
                    for track_id, cx, (x, y, fw, fh), penalty in detected_tracks:
                        current_ids.add(track_id)
                        
                        # 初見のIDなら初期化
                        if track_id not in self.person_scores:
                            self.person_scores[track_id] = 0.0
                        
                        # ペナルティ加算
                        # (顔が見えているなら penalty分、見えていない時間は加算されない...
                        #  だと寝ている人が有利になるので、「画面から消えた＝寝た」とみなすロジックが必要だが、
                        #  トラッキングモードでは「消えた＝見失った」なので難しい。
                        #  ここではシンプルに「映っている間の態度」を評価する)
                        self.person_scores[track_id] += penalty

                        # 顔画像を保存（最新のものに更新）
                        x0, y0 = max(0, y), max(0, x)
                        x1, y1 = min(frame.shape[0], y+fh), min(frame.shape[1], x+fw)
                        self.person_faces[track_id] = frame[x0:x1, y0:y1].copy()

                    # ------------------------------------
                    # 画面表示用の「ワースト3」を作成
                    # ------------------------------------
                    # スコアが高い順にソート: [(id, score), ...]
                    sorted_people = sorted(self.person_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # 上位3人のスコアをWeb表示用リストに入れる
                    # 0番目=ワースト1位, 1番目=ワースト2位...
                    self.display_ng_counts = [0, 0, 0]
                    for i in range(min(3, len(sorted_people))):
                        self.display_ng_counts[i] = int(sorted_people[i][1])

                else:
                    # ------------------------------------
                    # 結果発表フェーズ
                    # ------------------------------------
                    if self.winner == -1:
                        if not self.person_scores:
                            self.reason = "NOBODY DETECTED"
                            self.winner = -1
                        else:
                            # 最もスコアが高いIDを探す
                            worst_id, max_score = max(self.person_scores.items(), key=lambda x: x[1])
                            
                            if max_score > 0:
                                self.reason = f"WORST ID: {worst_id}"
                                # 便宜上、画面の「左(0)」を勝者として扱い、そこにワースト1位の顔を出す
                                self.winner = 0 
                                
                                # ワースト1位の顔画像を保存
                                if worst_id in self.person_faces:
                                    cv2.imwrite(FACE_PATH, self.person_faces[worst_id])
                            else:
                                self.winner = random.choice([0, 1, 2])
                                self.reason = "ALL GOOD (Random)"
                                # ダミー画像など保存処理（省略可）

                        print(f"★ 結果確定! {self.reason} ★")

                    self.analyzing = False

    def to_dict(self):
        with self.lock:
            # 画面側には「トップ3のスコア」を渡す
            # index.html では "左の人" が "ワースト1位" として表示されることになる
            return {
                "analyzing": self.analyzing,
                "remaining_time": int(self.remaining),
                "ng_counts": self.display_ng_counts, 
                "selected_person": self.winner,
                "reason": self.reason,
            }

    def get_face_image(self):
        # 変更なし
        if not os.path.exists(FACE_PATH):
            blank = np.zeros((260, 260, 3), dtype=np.uint8)
            cv2.imwrite(FACE_PATH, blank)
        return send_file(FACE_PATH, mimetype="image/jpeg")

    def cleanup(self):
        pass

state = State()