import numpy
def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	# Your code here
	return numpy.where(z < 0,z * alpha, z)
