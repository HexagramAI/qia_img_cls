import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compare_similarity(arr1, arr2):
    simiScore = cosine_similarity(arr1, arr2)
    return simiScore
