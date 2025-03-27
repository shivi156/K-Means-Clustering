import numpy as np

class KMeansClustering:
    def __init__(self, k=5, max_iterations=300):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    @staticmethod
    # This static method is used to calculate the Euclidean distance between each data point and centroids.
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def initialise_centroids(self, x):
        self.centroids = np.random.uniform(np.amin(x, axis=0),
                                           np.amax(x, axis=0),
                                           size=(self.k, x.shape[1]))

    def assign_clusters(self, x, max_iterations):
        for i in range(max_iterations):
            y = []
            for data_point in x:
                distances = self.euclidean_distance(data_point, self.centroids)
                cluster = np.argmin(distances)
                y.append(cluster)
            y = np.array(y)

    def fit(self, x, max_iterations=300):

        self.initialise_centroids(x)

        for i in range(max_iterations):
            y = []

            for data_point in x:
                # This will return a list of distances with a data point and all the centroids
                distances = self.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)

        # Repositioning centroids
            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(x[indices], axis=0)[0])
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0000000001:
                break
            else:
                self.centroids = np.array(cluster_centers)
        return y
