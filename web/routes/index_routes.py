from flask import Blueprint, render_template

index_routes = Blueprint("index", __name__)

@index_routes.route("/")
def home():
    return render_template("index.html")
