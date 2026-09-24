import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    A = A.astype(float)
    rank = len(A)

    # Your code here
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            if A[i][i] == 0:
                A[j] -= A[j][i]
            else:
                hs = A[j][i] / A[i][i]
                A[j] -= A[i] * hs
            

    for i in range(len(A)):
        if (np.abs(A[i]) < tol).all():
            rank -= 1

    return rank