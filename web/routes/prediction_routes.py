from flask import Blueprint, render_template

prediction_routes = Blueprint("prediction", __name__)

@prediction_routes.route("/")
def prediction():
    return render_template("prediction.html")
