def check_rules(invoice):
    reasons = []
    risk_score = 0

    # Rule 1: Unusual amount
    if invoice['amount'] > 2 * invoice['past_avg_amount']:
        reasons.append("Unusual high amount")
        risk_score += 25

    # Rule 2: Low trust vendor
    if invoice['trust_score'] < 40:
        reasons.append("Low trust vendor")
        risk_score += 25

    # Rule 3: Fraud history
    if invoice['past_fraud_count'] > 2:
        reasons.append("Vendor has fraud history")
        risk_score += 25

    return risk_score, reasons
