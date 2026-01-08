import face_recognition
import os
import pickle

# 設定
STUDENTS_DIR = "students"
DATA_FILE = "known_faces.pkl"  # 数値データの保存先

def create_face_data():
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(STUDENTS_DIR):
        print(f"エラー: '{STUDENTS_DIR}' フォルダが見つかりません。")
        return

    print("写真を読み込んで数値データに変換しています...")

    files = os.listdir(STUDENTS_DIR)
    count = 0

    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(STUDENTS_DIR, filename)
            try:
                # 画像を読み込み
                img = face_recognition.load_image_file(path)
                # 顔の特徴量（128次元のベクトル）を計算
                encs = face_recognition.face_encodings(img)
                
                if encs:
                    # 1人目の顔データを取得
                    known_face_encodings.append(encs[0])
                    
                    # ファイル名から名前を取得 (例: tanaka_01.jpg -> tanaka)
                    base_name = os.path.splitext(filename)[0]
                    name = base_name.split('_')[0]
                    known_face_names.append(name)
                    
                    count += 1
                    print(f"変換完了: {name} ({filename})")
                else:
                    print(f"警告: 顔が検出されませんでした -> {filename}")

            except Exception as e:
                print(f"エラースキップ: {filename} ({e})")

    # データをファイルに保存 (pickle形式)
    print(f"データを '{DATA_FILE}' に保存中...")
    with open(DATA_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print(f"\n完了！ {count} 人のデータを登録しました。")
    print(f"'{STUDENTS_DIR}' フォルダの中身は削除しても大丈夫です（バックアップは取っておいてください）。")

if __name__ == "__main__":
    create_face_data()
