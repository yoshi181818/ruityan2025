import cv2
import os
import numpy as np
import face_recognition
from state import state

# --- 顔認証データベースの準備 ---
known_face_encodings = []
known_face_names = []
STUDENTS_DIR = "students"

def load_known_faces():
    global known_face_encodings, known_face_names
    if not os.path.exists(STUDENTS_DIR):
        os.makedirs(STUDENTS_DIR)
        return

    print("【顔認証】データを読み込み中...")
    for filename in os.listdir(STUDENTS_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(STUDENTS_DIR, filename)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_face_encodings.append(encs[0])
                    # "_"で区切って名前を決める
                    base_name = os.path.splitext(filename)[0]
                    name = base_name.split('_')[0]
                    known_face_names.append(name)
                    print(f"登録: {name}")
            except Exception as e:
                print(f"Skip: {filename}")
    print(f"完了: {len(known_face_names)}人を登録")

load_known_faces()

# --- よそ見判定 ---
def calculate_penalty_landmarks(landmarks):
    try:
        left_eye = np.mean(landmarks['left_eye'], axis=0)
        right_eye = np.mean(landmarks['right_eye'], axis=0)
        nose = np.mean(landmarks['nose_tip'], axis=0)
        
        eye_width = np.linalg.norm(left_eye - right_eye)
        if eye_width == 0: return 0.0
        
        eyes_center = (left_eye + right_eye) / 2.0
        diff = abs(nose[0] - eyes_center[0]) / eye_width

        if diff > 0.4: return 0.5
        elif diff > 0.2: return 0.2
        return 0.0
    except:
        return 0.0

def process_frame(frame):
    # エラーで止まらないようにtryで囲む
    try:
        h, w, _ = frame.shape
        
        # 高速化のため縮小
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)
        lms = face_recognition.face_landmarks(rgb_small, locs)

        detected_people = []

        for (top, right, bottom, left), enc, lm in zip(locs, encs, lms):
            # 名前特定
            matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                first_index = matches.index(True)
                name = known_face_names[first_index]

            # 態度判定
            penalty = calculate_penalty_landmarks(lm)

            # 座標を元に戻す
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            # ★安全対策: 画面外にはみ出さないように座標を制限する
            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)
            
            fw = right - left
            fh = bottom - top

            detected_people.append((name, (left, top, fw, fh), penalty))

            # 描画
            color = (0, 0, 255) if name == "Unknown" else ((0, 255, 255) if penalty > 0 else (0, 255, 0))
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # 文字を描画（エラーになりやすいのでここもtryで囲む）
            try:
                cv2.putText(frame, str(name), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except Exception as e:
                print(f"文字描画エラー: {e}")

        # 集計へ
        state.update_logic_recognition(detected_people, frame)

    except Exception as e:
        # 万が一エラーが起きても、サーバーを止めずにエラー内容を表示するだけにする
        print(f"Critical Error in process_frame: {e}")

    return frame