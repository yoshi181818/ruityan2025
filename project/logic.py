import cv2
import mediapipe as mp
from state import state

mp_face = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5
)

latest_faces = [None,None,None]

def process_frame(frame):
    area_w = frame.shape[1] // 3
    ih, iw = frame.shape[0], frame.shape[1]

    # AI用
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(frame_rgb)

    faces = []
    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x = int(box.xmin * iw)
            y = int(box.ymin * ih)
            w = int(box.width * iw)
            h = int(box.height * ih)
            cx = x + w//2
            faces.append((cx, (x,y,w,h)))

    faces.sort(key=lambda f: f[0])
    current = [0,0,0]

    # 顔保存
    for cx,(x,y,w,h) in faces:
        x0=max(0,x); y0=max(0,y)
        x1=min(iw,x+w); y1=min(ih,y+h)
        crop = frame[y0:y1, x0:x1].copy()

        if cx < area_w:
            current[0] = 1; latest_faces[0] = crop
        elif cx < area_w*2:
            current[1] = 1; latest_faces[1] = crop
        else:
            current[2] = 1; latest_faces[2] = crop

        cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),2)

    # 判定ロジック
    state.update_logic(current, latest_faces, frame, area_w)

    return frame
