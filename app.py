import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from flask_cors import CORS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
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

        # Use a better prompt approach instead of list-based validation
        # Let the AI model itself determine educational relevance
        
        ai_validation_prompt = (
            f"You are an educational content validator. Analyze if '{topic}' is an educational topic related to "
            f"any field of academic study, including K-12 subjects, college courses, technical topics, "
            f"professional education, vocational training, or any legitimate area of learning.\n\n"
            f"Examples of valid educational topics include:\n"
            f"- Standard school subjects (math, science, history, languages)\n"
            f"- Engineering fields (CSE, ECE, mechanical, civil)\n"
            f"- Programming languages (Java, Python, JavaScript)\n"
            f"- Medical subjects (anatomy, cardiology, nursing)\n"
            f"- Intermediate/high school combinations (MPC, BiPC)\n"
            f"- Professional fields (accounting, management, law)\n"
            f"- Technical skills (cybersecurity, cloud computing)\n\n"
            f"Respond with only 'VALID' or 'INVALID'. If there's any educational merit to the topic, err on the side of considering it valid."
        )

        # First validate the topic
        validation_response = client.chat(
            model="mistral-tiny",
            messages=[ChatMessage(role="user", content=ai_validation_prompt)]
        )
        
        validation_result = validation_response.choices[0].message.content
        
        if "INVALID" in validation_result.upper():
            return jsonify({"error": "❌ This topic is not related to academic studies. Please enter a valid educational topic."}), 400

        # Now generate the study content with an enhanced prompt
        study_prompt = (
            f"You are an expert educator specializing in '{topic}'. Create a comprehensive, detailed, and academically rigorous "
            f"study guide that would be suitable for a student studying this subject.\n\n"
            f"Structure your response with these sections:\n\n"
            f"## 1. Introduction to {topic}\n"
            f"- Define the subject and its importance in the field\n"
            f"- Provide historical context if relevant\n"
            f"- Explain where this topic fits within the broader discipline\n\n"
            f"## 2. Core Concepts and Principles\n"
            f"- Explain the fundamental theories and ideas\n"
            f"- Break down complex topics into understandable components\n"
            f"- Include relevant formulas, diagrams, or frameworks\n\n"
            f"## 3. Key Topics and Sub-fields\n"
            f"- Explore the main areas of study within this topic\n"
            f"- Highlight important relationships between concepts\n\n"
            f"## 4. Practical Applications\n"
            f"- Describe real-world uses and implementations\n"
            f"- Include examples of how this knowledge is applied\n\n"
            f"## 5. Advanced Topics and Current Research\n"
            f"- Introduce more complex aspects of the subject\n"
            f"- Mention current trends and developments\n\n"
            f"## 6. Study Questions and Practice Problems\n"
            f"- Provide thought-provoking questions to test understanding\n"
            f"- Include example problems (with brief solutions) if applicable\n\n"
            f"## 7. Further Resources\n"
            f"- Suggest types of materials for additional study\n\n"
            f"Make the content academically rigorous but accessible to a motivated student."
        )

        # Generate notes using Mistral AI
        study_response = client.chat(
            model="mistral-tiny",
            messages=[ChatMessage(role="user", content=study_prompt)]
        )

        if study_response and study_response.choices:
            notes = study_response.choices[0].message.content
        else:
            return jsonify({"error": "⚠️ No response from Mistral AI"}), 500

        # Convert notes to a PDF
        pdf_filename = f"{topic.replace(' ', '_')}.pdf"
        pdf_path = os.path.join("generated_pdfs", pdf_filename)

        os.makedirs("generated_pdfs", exist_ok=True)  # Ensure directory exists
        create_pdf(pdf_path, topic, notes)  # Function to generate PDF

        return jsonify({
            "topic": topic,
            "notes": notes,
            "pdf_download_url": f"http://127.0.0.1:5000/download/{pdf_filename}"
        })

    except Exception as e:
        return jsonify({"error": f"🚨 An error occurred: {str(e)}"}), 500

def create_pdf(pdf_path, title, content):
    """Creates a PDF file with the given title and content."""
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, title)
    
    c.setFont("Helvetica", 12)
    y_position = 730

    for line in content.split("\n"):
        if y_position < 50:  # Avoid writing at the very bottom
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 750
        c.drawString(100, y_position, line)
        y_position -= 20  # Move down for next line

    c.save()

@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    """Allows users to download the generated PDF."""
    pdf_path = os.path.join("generated_pdfs", filename)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    return jsonify({"error": "❌ File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
