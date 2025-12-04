from flask import Flask, Response, render_template, jsonify
from camera import generate_stream
from state import state
import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
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
    return state.get_face_image()

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        state.cleanup()
