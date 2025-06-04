from sklearn.tree import _tree
import joblib

# Load model and encoders
clf = joblib.load("model/fraud_decision_tree.pkl")
encoders = joblib.load("model/label_encoders.pkl")

# Feature names (must match training)
features = ['transactionAmount', 'authentication', 'processingChannel',
            'merchantCategoryCode', 'accessChannel', 'deliveryChannelId',
            'scamFlag', 'typeOfLoss', 'transactionHour']

def tree_to_js(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined" for i in tree_.feature]

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            return (
                f"{indent}if (features['{name}'] <= {threshold}) {{\n"
                f"{recurse(tree_.children_left[node], depth + 1)}"
                f"{indent}}} else {{\n"
                f"{recurse(tree_.children_right[node], depth + 1)}"
                f"{indent}}}\n"
            )
        else:
            value = tree_.value[node][0]
            decision = int(value[1] > value[0])
            return f"{indent}return {decision};  // {value.tolist()} => {'FRAUD' if decision == 1 else 'NON-FRAUD'}\n"

    return "function isFraud(features) {\n" + recurse(0, 1) + "}\n"

# Generate JavaScript
js_code = tree_to_js(clf, features)

# Save to file
with open("fraud_rules.js", "w") as f:
    f.write(js_code)

print("✅ JavaScript rule function saved to fraud_rules.js")
