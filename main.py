import matplotlib.pyplot as plt
from kmeansclustering import KMeansClustering
from dataSet import DataSet

def plot_results(x, labels, centroids):
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('K-Means Clustering on Mall Customer Segmentation Data')
    plt.show()


def main():
    data_set = DataSet('Mall_Customers.csv', [3,4])
    x = data_set.load_data()
    kmeans = KMeansClustering(k=5)
    labels = kmeans.fit(x)
    plot_results(x, labels, kmeans.centroids)


if __name__ == "__main__":
        main()
