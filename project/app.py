from flask import Flask, Response, render_template, jsonify, request, redirect, session
from camera import generate_stream
from state import state
import time
from functools import wraps

app = Flask(__name__)
app.secret_key = "2411"  # 必ず変更！

# ★ ログインで入力させるパスワード（ここを変えるだけでOK）
LOGIN_PASSWORD = "yoshi1818"


# ---------------------------------------------------
# ログイン必須デコレータ
# ---------------------------------------------------
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect("/login")
        return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------
# Login Page
# ---------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        pw = request.form.get("password")

        if pw == LOGIN_PASSWORD:
            session["logged_in"] = True
            return redirect("/")
        else:
            return render_template("login.html", error="パスワードが違います")

    return render_template("login.html", error=None)


# ---------------------------------------------------
# Logout
# ---------------------------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------------------------------------------------
# メインUI（ログインしないと見れない）
# ---------------------------------------------------
@app.route("/")
@login_required
def index():
    return render_template("index.html")


# ---------------------------------------------------
# API（start / status / video_feed / face_image 全部ログイン必須）
# ---------------------------------------------------
@app.route("/start", methods=["POST"])
@login_required
def start_measure():
    state.start_time = time.time()
    state.reset()
    return ("", 204)


@app.route("/status")
@login_required
def status():
    return jsonify(state.to_dict())


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/face_image")
@login_required
def face_image():
    return state.get_face_image()


# ---------------------------------------------------
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        state.cleanup()
