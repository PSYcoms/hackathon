import pandas as pd
import numpy as np
import uuid
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load dataset
df = pd.read_csv("labeled_fraud_transaction_dataset.csv")

# Step 2: Drop high-cardinality and timestamp fields
df_model = df.drop(columns=["deviceInfo", "transactionTimestamp"])

# Step 3: Encode all categorical fields
label_encoders = {}
for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Step 4: Split features and labels
X = df_model.drop(columns=["label"])
y = df_model["label"]

# Step 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Decision Tree
clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Evaluate model
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Rule extraction logic
def extract_rules_from_tree(tree, feature_names, node_index=0, depth=0):
    indent = "  " * depth
    js_code = ""

    left = tree.children_left[node_index]
    right = tree.children_right[node_index]
    threshold = tree.threshold[node_index]
    feature = tree.feature[node_index]
    value = tree.value[node_index]

    # Leaf node
    if left == _tree.TREE_LEAF:
        prediction = int(np.argmax(value[0]))
        js_code += f"{indent}return {prediction};\n"
        return js_code

    # Non-leaf node
    feature_name = feature_names[feature]
    js_code += f"{indent}if (features['{feature_name}'] <= {threshold:.6f}) {{\n"
    js_code += extract_rules_from_tree(tree, feature_names, left, depth + 1)
    js_code += f"{indent}}} else {{\n"
    js_code += extract_rules_from_tree(tree, feature_names, right, depth + 1)
    js_code += f"{indent}}}\n"

    return js_code

# Step 9: Convert tree to JavaScript rule function
feature_names = list(X.columns)
js_function = "function detectFraud(features) {\n"
js_function += extract_rules_from_tree(clf.tree_, feature_names)
js_function += "}"

# Step 10: Save to JS file
with open("generated_fraud_rule.js", "w") as f:
    f.write(js_function)

print("✅ JavaScript rule saved to generated_fraud_rule.js")