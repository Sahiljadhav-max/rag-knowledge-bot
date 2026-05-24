from flask import Flask
from flask_cors import CORS
from app.config import Config

def create_app():
    app = Flask(__name__, static_folder="../frontend")
    app.config.from_object(Config)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    CORS(app)   # allow requests from the local frontend

    from app.routes import bp
    app.register_blueprint(bp)

    return app