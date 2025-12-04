from flask import Flask, Response, render_template, jsonify, request
from camera import generate_stream
from state import state
import time
from functools import wraps

app = Flask(__name__)

# ★ 管理用パスワード（必ず変えてね）
ADMIN_PASSWORD = "yoshi1818"


# ---- 認証デコレータ ----
def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        password = request.headers.get("X-ADMIN")
        if password != ADMIN_PASSWORD:
            return jsonify({"error": "unauthorized"}), 401
        return func(*args, **kwargs)
    return wrapper


# ---- IP制限（LAN内からだけアクセス許可など）----
@app.before_request
def limit_remote_addr():
    # 例：192.168.*.* だけ許可（自宅LAN想定）
    allowed_prefixes = ["192.168.", "10.", "172.16."]
    ip = request.remote_addr or ""
    if not any(ip.startswith(p) for p in allowed_prefixes) and ip != "127.0.0.1":
        return "Access Denied", 403


# ---- ルーティング ----
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
@require_auth
def start_measure():
    state.start_time = time.time()
    state.reset()
    return ("", 204)


@app.route("/status")
def status():
    return jsonify(state.to_dict())


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/face_image")
def face_image():
    # 顔画像取得（IP制限のみ。必要なら認証も追加してOK）
    return state.get_face_image()


if __name__ == "__main__":
    try:
        # ★ HTTPS化したいときは ssl_context を追加：
        # app.run(host="0.0.0.0", port=5000,
        #         ssl_context=("server.crt", "server.key"),
        #         debug=False, threaded=True)
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        state.cleanup()
