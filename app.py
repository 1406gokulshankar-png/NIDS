# app.py
# Purpose: the Flask web app (dashboard + JSON API for alerts).

import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
from sqlalchemy import func
from database import db, Alert

# Folder where this file lives (so the database always ends up in the project root)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# The only values allowed for an alert's status
VALID_STATUS = ["New", "Acknowledged", "False Positive"]

# We always report these names, even when their count is 0
ATTACK_TYPES = ["DoS", "Probe", "R2L", "U2R", "Unknown"]
SEVERITIES = ["Low", "Medium", "High", "Critical"]


def create_app():
    app = Flask(__name__)

    # Tell Flask where the SQLite file is. It is created if it doesn't exist.
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "nids.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # Create the tables if they don't exist yet
    with app.app_context():
        db.create_all()

    # ---- Home: send the visitor to the dashboard ----
    @app.route("/")
    def home():
        return redirect(url_for("dashboard"))

    # ---- The dashboard page (the HTML lives in templates/dashboard.html) ----
    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    # ---- List alerts (newest first, max 50). Optional filter: ?status=New ----
    @app.route("/api/alerts")
    def list_alerts():
        query = Alert.query
        status = request.args.get("status")
        if status in VALID_STATUS:
            query = query.filter_by(status=status)
        alerts = query.order_by(Alert.id.desc()).limit(50).all()
        return jsonify([a.to_dict() for a in alerts])

    # ---- Counts for the cards and charts ----
    @app.route("/api/stats")
    def stats():
        def count_by(column, names):
            # Ask the database: how many alerts per value of this column?
            rows = db.session.query(column, func.count(Alert.id)).group_by(column).all()
            counts = {name: 0 for name in names}      # start every name at 0
            for value, number in rows:
                counts[value] = number
            return counts

        return jsonify({
            "total": Alert.query.count(),
            "by_type": count_by(Alert.attack_type, ATTACK_TYPES),
            "by_severity": count_by(Alert.severity, SEVERITIES),
            "by_status": count_by(Alert.status, VALID_STATUS),
        })

    # ---- Change one alert's status (used by the buttons) ----
    @app.route("/api/alerts/<int:alert_id>/status", methods=["POST"])
    def update_status(alert_id):
        data = request.get_json(silent=True) or {}
        new_status = data.get("status")

        if new_status not in VALID_STATUS:
            return jsonify({"error": f"status must be one of {VALID_STATUS}"}), 400

        alert = db.session.get(Alert, alert_id)
        if alert is None:
            return jsonify({"error": "alert not found"}), 404

        alert.status = new_status
        db.session.commit()
        return jsonify(alert.to_dict())

    return app


app = create_app()

# This block runs only when we type:  python app.py
if __name__ == "__main__":
    app.run(debug=True)