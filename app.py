# app.py
# Purpose: the Flask web app (dashboard + JSON API for alerts).

import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
from sqlalchemy import func
from database import db, Alert, ReplayLog        # NEW: ReplayLog added
from evaluation import load_evaluation            # NEW

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

    # Create the tables if they don't exist yet (this also creates the new replay_log table)
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

    # ---- Explain one alert with SHAP (used by the Explain button) ----
    @app.route("/api/alerts/<int:alert_id>/explain")
    def explain_alert(alert_id):
        alert = db.session.get(Alert, alert_id)
        if alert is None:
            return jsonify({"error": "alert not found"}), 404
        if alert.test_row_index is None:
            return jsonify({"error": "this alert has no test row to explain"}), 400

        # Imported here so the dashboard starts fast; SHAP loads on first use
        from explain import explain_row
        result = explain_row(alert.test_row_index)
        result["alert"] = alert.to_dict()
        return jsonify(result)

    # ---- NEW (Feature 1): novel-attack evaluation numbers (written by replay.py) ----
    @app.route("/api/evaluation")
    def api_evaluation():
        data = load_evaluation()
        if data is None:
            return jsonify({"error": "No evaluation yet. Run replay.py first."}), 404
        return jsonify(data)

    # ---- NEW (Feature 2): concept drift report.
    #      Optional: /api/drift?last=500 uses only the newest 500 replayed rows ----
    @app.route("/api/drift")
    def api_drift():
        last = request.args.get("last", type=int)

        query = ReplayLog.query.order_by(ReplayLog.id.desc())
        if last:
            query = query.limit(last)
        row_indices = [log.test_row_index for log in query.all()]

        # Imported here so the app starts fast; the data loads on first use
        from drift import compute_drift
        return jsonify(compute_drift(row_indices))

    return app


app = create_app()

# This block runs only when we type:  python app.py
if __name__ == "__main__":
    app.run(debug=True)