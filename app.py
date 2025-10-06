import os
from flask import Flask, request, jsonify, render_template

from openai import OpenAI
from dotenv import load_dotenv
import base64

load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- SYSTEM PROMPT ----------
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
   - Lighting: small fix (move 0.5–1 m, rotate body/camera X°, use reflector/flag), plus a quick white balance note.
4) Quick-fix checklist — 4 very short checkboxes the user can try on the next shot.

Tone: supportive, specific, actionable. Avoid heavy beauty edits; favor authentic, joyful moments.
"""
# -----------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_photo():
    style = request.form.get("style", "natural candid")
    file = request.files.get("photo")
    if not file:
        return jsonify({"error": "No photo uploaded"}), 400

    # Convert to base64 data URL
    b64 = base64.b64encode(file.read()).decode("utf-8")
    mime = file.mimetype or "image/jpeg"
    data_url = f"data:{mime};base64,{b64}"

    # Put your system prompt in `instructions` for the Responses API
    system_prompt = SYSTEM_PROMPT.replace("{{STYLE}}", style)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",                  # vision-capable model
            instructions=system_prompt,           # <-- system prompt goes here
            input=[                               # type: ignore
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Please analyze this portrait for the '{style}' style."
                        },
                        {
                            "type": "input_image",
                            "image_url": data_url
                        }
                    ]
                }
            ]
        )

        # Use the SDK helper to get plain text
        text = response.output_text

        return jsonify({"style": style, "tips": text})

    except Exception as e:
        # Helpful error back to the client + server log
        print("OpenAI error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)