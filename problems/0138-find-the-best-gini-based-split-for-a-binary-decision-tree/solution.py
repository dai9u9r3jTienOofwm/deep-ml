import numpy as np
from typing import Tuple
def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	total = len(y)
	unique, counts = np.unique(y, return_counts=True)
	val = 1 - sum([(counts[i]/total)**2 for i in range(len(counts))])
	return round(val,3)

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    n_samples, n_features = X.shape
    best_gini = 2
    best_feature_index = -1
    best_threshold = 0
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        thresholds = np.unique(feature_values)

        # 2. Quét qua từng threshold
        for threshold in thresholds:
            # Chia nhánh bằng boolean indexing của NumPy (nhanh, không cần vòng lặp i)
            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold

            y_left = y[left_mask]
            y_right = y[right_mask]
            
            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)
            weighted_gini = (len(y_left) / n_samples) * gini_left + (len(y_right) / n_samples) * gini_right


            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_index = feature_idx
                best_threshold = float(threshold)

    return (best_feature_index, best_threshold)

        