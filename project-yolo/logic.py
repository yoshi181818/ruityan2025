import cv2
from ultralytics import YOLO
from state import state

# 顔専用モデル
model = YOLO("yolov8n-face.pt")

def calculate_penalty(box, keypoints):
    """
    顔の向きからペナルティ（NG度）を計算
    """
    if keypoints is None: return 0.0
    kps = keypoints.xy[0].cpu().numpy()
    if len(kps) < 3: return 0.0

    # 0:右目, 1:左目, 2:鼻
    right_eye_x = kps[0][0]
    left_eye_x = kps[1][0]
    nose_x = kps[2][0]

    eye_width = abs(left_eye_x - right_eye_x)
    if eye_width == 0: return 0.0

    eyes_center = (left_eye_x + right_eye_x) / 2.0
    diff = abs(nose_x - eyes_center) / eye_width

    # 厳しさ調整
    if diff > 0.5:
        return 0.5  # 完全よそ見
    elif diff > 0.25:
        return 0.2  # 少しよそ見
    else:
        return 0.0  # 正面

def process_frame(frame):
    h, w, _ = frame.shape

    # -----------------------------------------------------
    # ★変更点: model.track で追跡モードを有効にする (persist=True)
    # -----------------------------------------------------
    # tracker="bytetrack.yaml" は標準で組み込まれています
    results = model.track(frame, persist=True, verbose=False, conf=0.5)

    detected_tracks = []

    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints
        
        # 検出がない場合、boxesは空になることがある
        if boxes is None or boxes.id is None:
            continue

        # トラックIDを取得（IDがない場合はスキップ）
        track_ids = boxes.id.int().cpu().tolist()
        
        for i, box in enumerate(boxes):
            # ID取得
            track_id = track_ids[i]
            
            # 座標取得
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            fw = x2 - x1
            fh = y2 - y1
            cx = x1 + fw // 2
            
            # ペナルティ計算
            penalty = 0.0
            if keypoints is not None:
                # 該当するインデックスのキーポイントを渡す
                # (resultsが1人の場合と複数の場合で構造が変わらないよう処理)
                kps_item = r.keypoints[i]
                penalty = calculate_penalty(box, kps_item)

            # (ID, 中心X, 枠情報, ペナルティ)
            detected_tracks.append((track_id, cx, (x1, y1, fw, fh), penalty))

    # -----------------------------------------------------
    # 状態更新へ渡す
    # -----------------------------------------------------
    # 今回はエリア分けせず、IDごとのリストをそのまま渡します
    # latest_faces は state 側で作るのでここでは枠描画用画像を渡す必要なし
    
    # 画面への描画（IDも表示）
    for track_id, cx, (x, y, fw, fh), penalty in detected_tracks:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + fw)
        y1 = min(h, y + fh)
        
        # よそ見なら黄色、正面なら緑
        color = (0, 255, 255) if penalty > 0 else (0, 255, 0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        
        # IDを表示
        label = f"ID:{track_id}"
        cv2.putText(frame, label, (x0, y0 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 状態更新 (IDベースのリストを渡す)
    state.update_logic_tracking(detected_tracks, frame)

    return frame