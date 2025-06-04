import json
import random
from datetime import datetime, timedelta

merchants = ["Amazon", "Walmart", "Starbucks", "Apple Store", "Target", "Best Buy", "Costco", "eBay", "Home Depot", "CVS"]
locations = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "San Francisco, CA", "Houston, TX", "Miami, FL", "Seattle, WA", "Boston, MA", "Dallas, TX", "Atlanta, GA"]
payment_methods = ["credit_card", "debit_card", "mobile_wallet", "paypal", "bank_transfer"]

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

start_date = datetime(2024, 6, 1)
end_date = datetime(2024, 6, 20)

data = []
for i in range(500):
    record = {
        "transaction_id": f"TXN{1000001 + i}",
        "user_id": f"USR{str(random.randint(1, 200)).zfill(4)}",
        "amount": round(random.uniform(5.0, 1000.0), 2),
        "currency": "USD",
        "timestamp": random_date(start_date, end_date).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "merchant": random.choice(merchants),
        "location": random.choice(locations),
        "device_id": f"DEV{random.randint(10000, 99999)}",
        "payment_method": random.choice(payment_methods),
        "is_fraud": False
    }
    data.append(record)

with open("synthetic_non_fraud_payloads_500.json", "w") as f:
    json.dump(data, f, indent=2)