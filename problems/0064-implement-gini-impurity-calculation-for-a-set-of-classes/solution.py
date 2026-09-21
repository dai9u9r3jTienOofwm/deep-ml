
import numpy as np

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