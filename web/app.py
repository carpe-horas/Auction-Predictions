import os
os.environ["DISPLAY"] = ":0"  # Tkinter GUI 비활성화
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask
from flask_cors import CORS
from routes.analysis_routes import analysis_routes 
from routes.prediction_routes import prediction_routes
from routes.index_routes import index_routes
from routes.qna_routes import qna_routes

app = Flask(__name__, static_folder="static", template_folder="templates")

CORS(app)


# Flask Config
app.config.from_object("web.config.Config")

# Blueprint 등록
app.register_blueprint(index_routes, url_prefix="/")
app.register_blueprint(analysis_routes, url_prefix="/analysis")  
app.register_blueprint(prediction_routes, url_prefix="/prediction")
app.register_blueprint(qna_routes, url_prefix="/qna")

if __name__ == "__main__":
    app.run(debug=True)
