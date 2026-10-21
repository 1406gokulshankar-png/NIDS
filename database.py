# database.py
# Purpose: describe the database table (the "Alerts" table).

from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

# 'db' is the object that talks to the database.
# We create it here and connect it to the Flask app in app.py.
db = SQLAlchemy()


class Alert(db.Model):
    """One row = one alert raised by the system."""
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)          # auto-numbered 1, 2, 3...
    timestamp = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),       # filled automatically
        nullable=False,
    )
    source_info = db.Column(db.String(200))               # e.g. "tcp / http"
    attack_type = db.Column(db.String(50), nullable=False)  # DoS, Probe, R2L, U2R, Unknown
    anomaly_score = db.Column(db.Float, nullable=False)   # higher = more suspicious
    severity = db.Column(db.String(20), default="Low")    # Low / Medium / High / Critical
    status = db.Column(db.String(20), default="New")      # New / Acknowledged / False Positive
    test_row_index = db.Column(db.Integer)                # row number in the test set

    def to_dict(self):
        """Turn this row into a plain dictionary (easy to send as JSON)."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source_info": self.source_info,
            "attack_type": self.attack_type,
            "anomaly_score": round(self.anomaly_score, 4),
            "severity": self.severity,
            "status": self.status,
            "test_row_index": self.test_row_index,
        }