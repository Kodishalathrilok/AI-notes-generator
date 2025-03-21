import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, send_from_directory
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from flask_cors import CORS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textwrap import wrap

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for frontend communication

# Read API key from .env
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("❌ MISTRAL_API_KEY is missing! Add it in the .env file.")

# Initialize Mistral Client
client = MistralClient(api_key=MISTRAL_API_KEY)

@app.route("/", methods=["GET"])
def home():
    """API Health Check"""
    return jsonify({"message": "✅ Mistral AI Note Generator API is running!"})

@app.route("/generate", methods=["POST"])
def generate():
    """Generate AI-powered detailed notes and convert them to PDF."""
    try:
        data = request.get_json()
        if not data or "topic" not in data:
            return jsonify({"error": "❌ Missing 'topic' in request body"}), 400

        topic = data.get("topic", "").strip()
        if not topic:
            return jsonify({"error": "❌ Topic cannot be empty"}), 400

        # Validate topic relevance
        validation_prompt = f"Is '{topic}' a valid academic topic? Reply only 'VALID' or 'INVALID'."
        validation_response = client.chat(
            model="mistral-tiny",
            messages=[ChatMessage(role="user", content=validation_prompt)]
        )

        if validation_response.choices and "INVALID" in validation_response.choices[0].message.content.upper():
            return jsonify({"error": "❌ Please enter a valid educational topic."}), 400

        # Generate AI Notes
        study_prompt = f"Generate detailed study notes for: {topic}"
        study_response = client.chat(
            model="mistral-tiny",
            messages=[ChatMessage(role="user", content=study_prompt)]
        )

        if not study_response.choices:
            return jsonify({"error": "❌ No response from AI."}), 500

        notes = study_response.choices[0].message.content.strip()

        # Convert notes to a PDF
        pdf_filename = f"{topic.replace(' ', '_')}.pdf"
        pdf_path = os.path.join("generated_pdfs", pdf_filename)
        os.makedirs("generated_pdfs", exist_ok=True)
        create_pdf(pdf_path, topic, notes)

        return jsonify({
            "topic": topic,
            "notes": notes,
            "pdf_download_url": f"/download/{pdf_filename}"
        })

    except Exception as e:
        return jsonify({"error": f"🚨 An error occurred: {str(e)}"}), 500

def create_pdf(pdf_path, title, content):
    """Creates a PDF file with the given title and content, ensuring proper text wrapping."""
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, title)
    
    c.setFont("Helvetica", 12)
    y_position = 730

    # Wrap text for better formatting
    wrapped_text = "\n".join(["\n".join(wrap(line, 80)) for line in content.split("\n")])

    for line in wrapped_text.split("\n"):
        if y_position < 50:  # Page break when reaching bottom margin
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 750
        c.drawString(100, y_position, line)
        y_position -= 20  

    c.save()

@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    """Allows users to download the generated PDF."""
    pdf_path = os.path.join("generated_pdfs", filename)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    return jsonify({"error": "❌ File not found"}), 404

### ✅ PWA Support ###
@app.route("/manifest.json")
def serve_manifest():
    return send_from_directory("static", "manifest.json")

@app.route("/service-worker.js")
def serve_service_worker():
    return send_from_directory("static", "service-worker.js")

if __name__ == "__main__":
    app.run(debug=True)
