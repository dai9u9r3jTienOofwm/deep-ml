import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores = np.array(scores)
    m = np.max(scores, axis=0,keepdims=True)
    scores -= m

    return list(np.exp(scores) / np.sum(np.exp(scores),axis=0,keepdims=True))