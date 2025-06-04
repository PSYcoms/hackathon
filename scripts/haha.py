import json
import random
import uuid
from datetime import datetime, timedelta

def random_id(length=10):
    return ''.join(random.choices('0123456789', k=length))

def random_date(start, end):
    return (start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))).isoformat() + "Z"

records = []
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 6, 1)

for i in range(500):
    record = {
        "data": {
            "attributes": {
                "payload": {
                    "sourceFeed": "SAS ETL",
                    "sourceSystem": "HADOOP",
                    "keys": [
                        {
                            "valueClassification": "PII",
                            "type": "businessCustomerId",
                            "value": {"value": random_id(), "type": "PLAINTEXT"}
                        },
                        {
                            "valueClassification": "PII",
                            "type": "creditAcctNumberOrIban",
                            "value": {"value": random_id(14), "type": "PLAINTEXT"}
                        },
                        {
                            "valueClassification": "PII",
                            "type": "creditCustomerId",
                            "value": {"value": random_id(), "type": "PLAINTEXT"}
                        },
                        {
                            "valueClassification": "PII",
                            "type": "customerAcctNumber",
                            "value": {"value": random_id(14), "type": "PLAINTEXT"}
                        },
                        {
                            "valueClassification": "PII",
                            "type": "customerId",
                            "value": {"value": random_id(), "type": "PLAINTEXT"}
                        },
                        {
                            "valueClassification": "PII",
                            "type": "jointCustomerId",
                            "value": {"value": random_id(), "type": "PLAINTEXT"}
                        }
                    ],
                    "changes": [
                        {
                            "newValue": {
                                "creditAcctSortCode": random.randint(100000, 999999),
                                "membershipId": random_id(12),
                                "deviceId": random_id(24),
                                "transactionTimestamp": random_date(start_date, end_date),
                                "loss": "Y",
                                "transactionCountryCode": str(random.randint(100, 999)),
                                "accessChannel": random.choice(["M", "U", "W"]),
                                "terminalRef": uuid.uuid4().hex[:16],
                                "reportedDate": random_date(start_date, end_date)[:10],
                                "transactionAmount": round(random.uniform(1000, 10000000), 2),
                                "stoFinalPaymentAmount": round(random.uniform(1000, 100000000), 2),
                                "barclaysDeviceId": {
                                    "tokenisedValue": uuid.uuid4().hex,
                                    "currentKeyAndSalt": {
                                        "keyName": "ATIR_TOKENISATION_KEY",
                                        "saltName": "SALT_DATA_ATTRIBUTE_LENGTH_60"
                                    }
                                },
                                "comparisonValues": [],
                                "scamFlag": "Y",
                                "authentication": random.choice(["Y", "N", "M"]),
                                "deliveryChannelId": random.choice(["A", "G"]),
                                "ipAddress": ".".join(str(random.randint(0, 255)) for _ in range(4)),
                                "sessionId": uuid.uuid4().hex[:20],
                                "transactionPostalCodeClr": str(random.randint(1000, 9999)),
                                "deviceInfo": "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
                                "fraudModsOperand": "MULTI_SAME_BENE_BTB",
                                "transactionType": "C",
                                "uvmItCallId": random_id(9),
                                "merchantCategoryCode": str(random.choice([5411, 6011, 8398])),
                                "tppId": uuid.uuid4().hex,
                                "typeOfLoss": "FRAUD_GEN",
                                "processingChannel": "A",
                                "interactionId": str(random.randint(100000, 9999999)),
                                "stoRecurringPaymentAmount": round(random.uniform(1000, 100000000), 2)
                            }
                        }
                    ],
                    "externalTransactionId": random_id(18)
                },
                "header": {
                    "upstreamSystemOutTimestamp": random_date(start_date, end_date),
                    "businessUnit": "B",
                    "fraudGatewaySystemInTimestamp": random_date(start_date, end_date),
                    "feedType": "RETAILCONFIRMEDFRAUD",
                    "upstreamSystemInTimestamp": random_date(start_date, end_date),
                    "fraudGatewaySystemOutTimestamp": random_date(start_date, end_date),
                    "uniqueId": str(uuid.uuid4()),
                    "eventTimestamp": random_date(start_date, end_date)
                }
            }
        }
    }
    records.append(record)

with open("mock_confirmed_fraud_payloads.json", "w") as f:
    json.dump(records, f, indent=2)