import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rules import check_rules

seen_invoices = set()

def check_duplicate(invoice_id):
    if invoice_id in seen_invoices:
        return True
    seen_invoices.add(invoice_id)
    return False


# Load dataset
data = pd.read_csv("fraud-detection-ai-model-/invoice_data.csv")


# Features (input)
X = data[['amount', 'trust_score', 'past_avg_amount', 'past_fraud_count']]

# Target (output)
y = data['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

import pickle
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully!")


# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Risk score
risk_score = model.predict_proba(X_test)[:,1]
print("Sample Risk Scores:", risk_score[:5])

# Manual test
sample = pd.DataFrame([[8000, 30, 3000, 3]],
                      columns=['amount', 'trust_score', 'past_avg_amount', 'past_fraud_count'])

invoice = {
    'invoice_id': 'INV001',
    'amount': 8000,
    'trust_score': 30,
    'past_avg_amount': 3000,
    'past_fraud_count': 3
}



print("Prediction:", model.predict(sample))
ml_score = model.predict_proba(sample)[0][1] * 100
print("ML Score:", ml_score)

rule_score, reasons = check_rules(invoice)

check_duplicate(invoice['invoice_id'])

if check_duplicate(invoice['invoice_id']):
    rule_score += 25
    reasons.append("Duplicate invoice detected")


print("Rule Score:", rule_score)
print("Reasons:", reasons)



final_score = (ml_score * 0.6) + (rule_score * 0.4)

print("Final Score:", final_score)

if final_score > 70:
    status = "HIGH RISK"
elif final_score > 40:
    status = "MEDIUM RISK"
else:
    status = "LOW RISK"

print("Risk Status:", status)

print("Second duplicate check:", check_duplicate(invoice['invoice_id']))
