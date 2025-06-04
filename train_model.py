from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Prepare features and labels
df = combined_df.copy()

# Drop rows with missing critical values
df = df.dropna(subset=["transactionAmount", "authentication", "processingChannel", "merchantCategoryCode"])

# Encode categorical columns
categorical_cols = ["authentication", "processingChannel", "merchantCategoryCode", 
                    "accessChannel", "deliveryChannelId", "scamFlag", "typeOfLoss"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Convert timestamps to numeric values (hour of transaction as a proxy)
df["transactionHour"] = pd.to_datetime(df["transactionTimestamp"]).dt.hour
df = df.drop(columns=["transactionTimestamp", "deviceInfo", "ipAddress", "transactionPostalCodeClr"])

# Separate features and target
X = df.drop(columns=["label"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree model
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Export the tree as human-readable rules
rules = export_text(clf, feature_names=list(X.columns))

rules_output_path = "/mnt/data/decision_tree_rules.txt"
with open(rules_output_path, "w") as f:
    f.write(rules)

rules_output_path
