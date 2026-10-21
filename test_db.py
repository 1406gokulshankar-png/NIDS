# test_db.py
# Purpose: check that the database works (add -> read -> delete one fake alert).

from app import app
from database import db, Alert

# Database actions must happen inside an "app context"
with app.app_context():
    # 1. Add a fake alert
    fake = Alert(
        source_info="tcp / http",
        attack_type="DoS",
        anomaly_score=0.82,
        severity="High",
        test_row_index=0,
    )
    db.session.add(fake)
    db.session.commit()
    print("Added alert with id:", fake.id)

    # 2. Read all alerts back
    print("Alerts in database:", Alert.query.count())
    print("First alert:", Alert.query.first().to_dict())

    # 3. Delete the fake alert so the database stays clean
    db.session.delete(fake)
    db.session.commit()
    print("After delete, alerts in database:", Alert.query.count())