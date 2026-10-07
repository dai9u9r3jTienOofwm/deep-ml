import numpy as np

def softmax(x: list[float]):
	x = np.array(x)
	max_value = np.max(x, axis = 0, keepdims=True)
	x -= max_value

	return np.exp(x)/ np.sum(np.exp(x), axis = 0, keepdims=True)

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	p = softmax(x)

	return np.diag(p) - np.outer(p,p)

	