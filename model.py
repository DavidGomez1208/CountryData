import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score


def preprocessing(df, features):
    X = df[features].copy()

    numerical = X.select_dtypes(include=["int64", "float64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns

    encoder = LabelEncoder()
    for column in categorical:
        X[column] = encoder.fit_transform(X[column].astype(str))

    scaler = StandardScaler()
    X[numerical] = scaler.fit_transform(X[numerical])

    if len(categorical) > 0:
        X[categorical] = scaler.fit_transform(X[categorical])

    return X


def pca_reduction(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def clustering(X, model_type, n_clusters=3, metric="euclidean",
               eps=0.5, min_samples=5, linkage="ward"):

    if model_type == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    elif model_type == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)

    elif model_type == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters,
                                        linkage=linkage)

    else:
        raise ValueError("Invalid clustering algorithm.")

    return model.fit_predict(X)


def evaluate(X, labels):
    # Ignore noise points from DBSCAN (label -1).
    valid = labels != -1
    X = X[valid]
    labels = labels[valid]

    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return None, None

    silhouette = silhouette_score(X, labels)
    davies = davies_bouldin_score(X, labels)

    return silhouette, davies
