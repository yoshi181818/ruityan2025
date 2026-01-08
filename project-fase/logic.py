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
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(STUDENTS_DIR):
        os.makedirs(STUDENTS_DIR)
        return

    print("【顔認証】データを読み込み中...")
    
    # ファイル数チェック
    all_files = [f for f in os.listdir(STUDENTS_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"フォルダ内の画像ファイル数: {len(all_files)} 枚")

    for filename in all_files:
        path = os.path.join(STUDENTS_DIR, filename)
        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            
            if encs:
                # 顔が見つかった場合
                base_name = os.path.splitext(filename)[0]
                name = base_name.split('_')[0]

                for enc in encs:
                    known_face_encodings.append(enc)
                    known_face_names.append(name)
                    print(f"⭕ 成功: {name} ({filename})")
            else:
                # ★ここが重要：顔が見つからなかった場合
                print(f"❌ 失敗: {filename}")

        except Exception as e:
            print(f"⚠️ エラー読み込み不可: {filename} ({e})")
    
    print(f"--------------------------------------------------")
    print(f"完了: {len(all_files)} 枚中、{len(known_face_names)} パターンの顔データを登録しました")
    
# 起動時に学習を実行
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

        # よそ見の基準（数値を小さくすると厳しくなる）
        if diff > 0.4: return 0.5
        elif diff > 0.2: return 0.2
        return 0.0
    except:
        return 0.0


def process_frame(frame):
    # エラーで止まらないようにtryで囲む
    try:
        h, w, _ = frame.shape
        
        # ★修正2: 画質向上 (0.25 -> 0.5)
        # 少し重くなりますが、認識精度が格段に上がります
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)
        lms = face_recognition.face_landmarks(rgb_small, locs)

        detected_people = []

        for (top, right, bottom, left), enc, lm in zip(locs, encs, lms):
            # ★修正3: 名前特定ロジックの強化（最短距離法）
            name = "Unknown"
            
            if len(known_face_encodings) > 0:
                # 全データとの「似てなさ（距離）」を計算
                face_distances = face_recognition.face_distance(known_face_encodings, enc)
                
                # 一番似ているデータのインデックスを取得
                best_match_index = np.argmin(face_distances)
                
                # 距離が 0.45 以下なら本人と認定（0.6だとガバガバ、0.4だと厳しすぎ）
                if face_distances[best_match_index] < 0.45:
                    name = known_face_names[best_match_index]

            # 態度判定
            penalty = calculate_penalty_landmarks(lm)

            # 座標を元に戻す (0.5倍にしたので、2倍に戻す)
            scale = 2  # 1 / 0.5
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            
            # 画面外にはみ出さないように制限
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
            
            try:
                # 名前とスコア（デバッグ用）を表示したい場合はここで
                label = f"{name}"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except Exception as e:
                pass

        # 集計へ
        state.update_logic_recognition(detected_people, frame)

    except Exception as e:
        print(f"Critical Error in process_frame: {e}")

    return frame
