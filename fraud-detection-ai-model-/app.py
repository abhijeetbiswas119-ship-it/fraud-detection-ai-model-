
from flask import Flask, request, jsonify
import pickle
import os
from rules import check_rules

app = Flask(__name__)


# Load saved model
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.pkl")
model = pickle.load(open(model_path, "rb"))


# Store invoice IDs for duplicate detection
seen_invoices = set()


@app.route("/")
def home():
    return "Invoice Fraud Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    invoice_id = data["invoice_id"]
    amount = data["amount"]
    trust_score = data["trust_score"]
    past_avg_amount = data["past_avg_amount"]
    past_fraud_count = data["past_fraud_count"]

    # ML prediction
    sample = [[amount, trust_score, past_avg_amount, past_fraud_count]]
    ml_prob = model.predict_proba(sample)[0][1] * 100

    # Rule engine
    invoice = {
        "amount": amount,
        "trust_score": trust_score,
        "past_avg_amount": past_avg_amount,
        "past_fraud_count": past_fraud_count
    }

    rule_score, reasons = check_rules(invoice)

    # Duplicate detection
    if invoice_id in seen_invoices:
        rule_score += 25
        reasons.append("Duplicate invoice detected")
    else:
        seen_invoices.add(invoice_id)

    # Final score
    final_score = (ml_prob * 0.6) + (rule_score * 0.4)

    # Risk level
    if final_score > 70:
        status = "HIGH RISK"
    elif final_score > 40:
        status = "MEDIUM RISK"
    else:
        status = "LOW RISK"

    return jsonify({
        "risk_score": round(final_score, 2),
        "status": status,
        "reasons": reasons
    })


if __name__ == "__main__":
    app.run(debug=True)
