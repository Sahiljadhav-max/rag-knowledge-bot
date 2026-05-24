import os
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app.config import Config
from app.ingest import ingest_document
from app.rag_engine import rag_engine

bp = Blueprint("api", __name__)

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@bp.route("/upload", methods=["POST"])
def upload():
    """Accepts a file, ingests it, reloads the RAG chain."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported type. Allowed: {Config.ALLOWED_EXTENSIONS}"
        }), 400

    filename = secure_filename(file.filename)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        stats = ingest_document(save_path)
        rag_engine.reload()              # refresh the in-memory chain
        return jsonify({"success": True, "stats": stats}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/chat", methods=["POST"])
def chat():
    """Accepts a question, returns an answer with source citations."""
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    result = rag_engine.query(question)
    return jsonify(result), 200

@bp.route("/documents", methods=["GET"])
def list_documents():
    """Returns list of all uploaded files."""
    folder = Config.UPLOAD_FOLDER
    if not os.path.exists(folder):
        return jsonify({"files": []})
    files = [f for f in os.listdir(folder) if allowed_file(f)]
    return jsonify({"files": files})

@bp.route("/reset", methods=["POST"])
def reset():
    """Clears the vector store and starts fresh."""
    import shutil
    path = Config.VECTOR_STORE_PATH
    if os.path.exists(path):
        shutil.rmtree(path)
    upload_path = Config.UPLOAD_FOLDER
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path)
    rag_engine.vector_store = None
    rag_engine.chain = None
    return jsonify({"success": True, "message": "Index cleared."})

@bp.route("/")
def index():
    return send_from_directory("../frontend", "index.html")