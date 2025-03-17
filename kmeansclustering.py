import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:

    # Setting the value of k by default to 3
    def __init__(self, k=3):
        self.k = k
        self.centroids = None
