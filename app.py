import os
import json
import time
import base64
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# --- Realtime video (WebRTC) pieces ---
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
import av
import cv2 as cv  # pip install opencv-python

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- PROMPT --------------------
SYSTEM_PROMPT = """
You are a professional portrait photographer and coach.
Task: Analyze the uploaded photo and give short, encouraging guidance to improve it.

User’s desired style: {{STYLE}}  (e.g., “natural candid,” “editorial moody,” “bright & airy,” “street vibe”)

Write in natural, friendly language. Prefer 1–2 sentences per bullet. Avoid jargon unless necessary.

Return sections in plain text:

1) Style fit — one sentence on how the current shot aligns with {{STYLE}} and the main gap.
2) For the model (pose/emotion):
   - 3 concise tips on posture, head/shoulders/hands, micro-expressions.
3) For the photographer (composition/params/light):
   - Composition: rule-of-thirds/leading lines/background cleanup/crop suggestion.
   - Camera: suggested focal length range, aperture, shutter/ISO starting point for this scene.
   - Lighting: small fix (move 0.5–1 m, rotate body/camera X°), plus a quick white balance note.
4) Quick-fix checklist — 4 very short checkboxes the user can try on the next shot.

Tone: supportive, specific, actionable. Avoid heavy beauty edits; favor authentic, joyful moments.
"""
# ------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_photo():
    """Manual photo upload → OpenAI vision tips."""
    style = request.form.get("style", "natural candid")
    file = request.files.get("photo")
    if not file:
        return jsonify({"error": "No photo uploaded"}), 400

    # Base64 data URL for the image
    b64 = base64.b64encode(file.read()).decode("utf-8")
    mime = file.mimetype or "image/jpeg"
    data_url = f"data:{mime};base64,{b64}"

    # Put the prompt into `instructions` for Responses API
    system_prompt = SYSTEM_PROMPT.replace("{{STYLE}}", style)

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            instructions=system_prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text",
                         "text": f"Please analyze this portrait for the '{style}' style."},
                        {"type": "input_image", "image_url": data_url}
                    ]
                }
            ]
        )
        return jsonify({"style": style, "tips": resp.output_text})
    except Exception as e:
        print("OpenAI error:", e)
        return jsonify({"error": str(e)}), 500


# ----------------- WebRTC realtime path -----------------

async def _handle_offer(offer_sdp: str, offer_type: str):
    """
    Receives a WebRTC offer from the browser, creates an RTCPeerConnection,
    reads incoming video frames, computes lightweight heuristics, and sends
    back JSON tips over a data channel named 'tips'.
    """
    pc = RTCPeerConnection()
    tips_channel = None

    @pc.on("datachannel")
    def on_datachannel(channel):
        nonlocal tips_channel
        # The browser creates the 'tips' channel; we use it to send guidance back
        tips_channel = channel

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            # discard audio (or any non-video) tracks
            MediaBlackhole().addTrack(track)
            return

        async def recv_loop():
            last_sent = 0.0
            while True:
                frame: av.VideoFrame = await track.recv()
                img = frame.to_ndarray(format="bgr24")  # HxWx3 BGR
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Heuristic 1: brightness (simple mean luminance)
                luma = float(gray.mean()) / 255.0

                # Heuristic 2: horizon-tilt proxy (Canny + Hough)
                edges = cv.Canny(gray, 80, 160)
                lines = cv.HoughLines(edges, 1, 3.14159 / 180, 120)
                tilt_deg = 0.0
                if lines is not None:
                    angs = []
                    for rho, theta in lines[:8, 0]:
                        deg = theta * 180 / 3.14159
                        # distance from horizontal
                        tilt = min(abs(deg - 0), abs(deg - 180), abs(deg - 360))
                        angs.append(tilt)
                    if angs:
                        tilt_deg = sum(angs) / len(angs)

                # Throttle: send ≈2 msgs/sec
                now = time.time()
                if tips_channel and (now - last_sent) > 0.5:
                    tip = (
                        "Nice base exposure."
                        if 0.45 <= luma <= 0.65
                        else ("Scene a bit dark—step closer to window/open shade or raise ISO."
                              if luma < 0.45
                              else "Highlights strong—turn 10–15° or find softer light.")
                    )
                    if tilt_deg > 7:
                        tip += " Horizon looks tilted; level the phone slightly."

                    tips_channel.send(json.dumps({
                        "tip": tip,
                        "luma": round(luma, 2),
                        "tilt_deg": round(tilt_deg, 1)
                    }))
                    last_sent = now

        asyncio.create_task(recv_loop())

    offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    await pc.setRemoteDescription(offer)

    # Create and return answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription


# Tiny route to silence favicon 404s
@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/offer", methods=["POST"])
def offer():
    """Signaling endpoint: browser posts SDP offer → we return SDP answer."""
    data = request.get_json(force=True)
    sdp = data["sdp"]
    typ = data["type"]

    # Create a fresh event loop for this thread to avoid
    # "There is no current event loop in thread" errors.
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        answer = loop.run_until_complete(_handle_offer(sdp, typ))
        return jsonify({"sdp": answer.sdp, "type": answer.type})
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# --------------------------------------------------------

if __name__ == "__main__":
    # Dev: http. For production use HTTPS (required by mobile browsers for camera/WebRTC).
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=False)