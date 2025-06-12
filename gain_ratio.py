import numpy as np
import math
from collections import Counter


def entropy(y):
    counter = Counter(y)
    total = len(y)
    return -sum((count / total) * math.log2(count / total) for count in counter.values())

def information_gain_ratio(y, y_left, y_right):
    total_entropy = entropy(y)
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    info_gain = total_entropy - (n_left / n) * entropy(y_left) - (n_right / n) * entropy(y_right)

    split_info = 0
    for subset in [y_left, y_right]:
        if len(subset) == 0:
            continue
        p = len(subset) / n
        split_info -= p * math.log2(p)

    if split_info == 0:
        return 0

    return info_gain / split_info

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class GainRatioDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(set(y))
        
        current_impurity = entropy(y)  # ← compute impurity here

        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            leaf = TreeNode(value=leaf_value)
            leaf.impurity = current_impurity  # ← store it in the node
            return leaf

        best_gain_ratio = -1
        best_feature, best_threshold = None, None
        best_splits = None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_idxs = X[:, feature_index] <= threshold
                right_idxs = ~left_idxs
                y_left, y_right = y[left_idxs], y[right_idxs]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain_ratio = information_gain_ratio(y, y_left, y_right)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature_index
                    best_threshold = threshold
                    best_splits = (left_idxs, right_idxs)

        if best_gain_ratio == -1:
            leaf_value = Counter(y).most_common(1)[0][0]
            leaf = TreeNode(value=leaf_value)
            leaf.impurity = current_impurity
            return leaf

        left = self._build_tree(X[best_splits[0]], y[best_splits[0]], depth + 1)
        right = self._build_tree(X[best_splits[1]], y[best_splits[1]], depth + 1)
        node = TreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)
        node.impurity = current_impurity  # ← store impurity here too
        return node


    def _predict(self, node, x):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(node.left, x)
        return self._predict(node.right, x)

    def predict(self, X):
        return np.array([self._predict(self.root, sample) for sample in X])
    
    def to_dict(self):
        def to_python_type(value):
            # Convert NumPy types to native Python types
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            return value

        def recurse(node):
            if node.is_leaf_node():
                value = to_python_type(node.value)
                impurity = to_python_type(getattr(node, "impurity", None))
                return {
                    "name": f"Leaf: {value}",
                    "feature": None,
                    "threshold": None,
                    "left": None,
                    "right": None,
                    "value": value,
                    "impurity": impurity
                }

            feature = to_python_type(node.feature)
            threshold = to_python_type(node.threshold)
            impurity = to_python_type(getattr(node, "impurity", None))

            return {
                "name": f"Feature {feature} <= {threshold:.3f}",
                "feature": feature,
                "threshold": threshold,
                "left": recurse(node.left),
                "right": recurse(node.right),
                "value": None,
                "impurity": impurity
            }

        return recurse(self.root)



