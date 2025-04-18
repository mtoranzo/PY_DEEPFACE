from flask import Flask
from modules.routes import blueprint


def create_app():
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    return app
