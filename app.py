import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model import preprocessing, pca_reduction, clustering, evaluate

st.title("Country Data Clustering")

# Data Input
file = st.file_uploader("Upload your dataset", type=["csv"])
if file is None:
    st.stop()

df = pd.read_csv(file)
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Feature Selection
st.subheader("Feature Selection")
available = [c for c in df.columns if c.lower() != "country"]
mode = st.selectbox("Feature mode", ["Manual Selection", "PCA Reduction"])

if mode == "Manual Selection":
    features = st.multiselect("Select at least 2 features", available,
                              default=available[:2])
    if len(features) < 2:
        st.warning("Select at least 2 features")
        st.stop()
    X = preprocessing(df, features)
else:
    X = preprocessing(df, available)
    n_components = st.slider("Number of components", 2,
                              min(X.shape[0], X.shape[1]), 2)
    X, _ = pca_reduction(X, n_components)

# Model Selection
st.subheader("Model Selection")
model_type = st.selectbox(
    "Select clustering algorithm",
    ["K-Means", "DBSCAN", "Hierarchical"]
)

n_clusters, metric, eps, min_samples, linkage = 3, "euclidean", 0.5, 5, "ward"

if model_type in ["K-Means", "K-Medoids", "Hierarchical"]:
    n_clusters = st.slider("n_clusters", 2, 10, 3)
if model_type == "K-Medoids":
    metric = st.selectbox("metric", ["euclidean", "manhattan"])
if model_type == "DBSCAN":
    eps = st.slider("eps", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.slider("min_samples", 2, 10, 5)
if model_type == "Hierarchical":
    linkage = st.selectbox("linkage", ["ward", "complete", "average", "single"])

# Clustering and Output
if st.button("Run clustering"):
    labels = clustering(X, model_type, n_clusters, metric,
                        eps, min_samples, linkage)
    df["cluster"] = labels

    st.subheader("Cluster Labels")
    st.dataframe(df[["country", "cluster"]] if "country" in df.columns
                 else df[["cluster"]])
    st.subheader("Clustered Dataset")
    st.dataframe(df)

    # Visualization
    st.subheader("Visualization")
    if mode == "Manual Selection" and len(features) == 2:
        plot_data = X.values
        x_label, y_label = features[0], features[1]
    else:
        plot_data, _ = pca_reduction(X, 2)
        x_label, y_label = "PC1", "PC2"

    fig, ax = plt.subplots()
    ax.scatter(plot_data[:, 0], plot_data[:, 1], c=labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    st.pyplot(fig)

    # Evaluation
    st.subheader("Evaluation")
    silhouette, davies = evaluate(X, labels)
    if silhouette is None:
        st.warning("Metrics cannot be calculated for these clusters.")
    else:
        st.write("Silhouette Score:", silhouette)
        st.write("Davies-Bouldin Index:", davies)
