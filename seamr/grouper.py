from sklearn import cluster


class ObservationGrouper:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = cluster.MiniBatchKMeans( n_clusters = n_clusters, batch_size=1000 )
    
    def fit(self, X):
        self.model.fit(X)
    
    def segment(self, X):

        groups = self.predict(X)
        indices = list(range(X.shape[0]))

        for g,items in groupby(zip(indices, groups), key=lambda T: T[1]):
            gIndices = [ i for i,_ in items ]
            yield X[gIndices, :]