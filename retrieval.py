import numpy as np
from sklearn.decomposition import PCA
from feature_combination import combine_features
from similarity import cosine_distance, cross_bin_dissimilarity, jensen_shannon_divergence, jaccard_similarity


def train_pca(database, selected_keys, n_components=256):
    """
    Train a PCA model on the combined feature vectors of the entire database.
    """
    feature_list = []
    for img_path, feat_dict in database.items():
        combined = combine_features(feat_dict, selected_keys)
        feature_list.append(combined)
    X = np.vstack(feature_list)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def search_similar(query_features_dict, database, selected_keys, top_k=10, similarity="Cosine", pca=None):
    query_combined = combine_features(query_features_dict, selected_keys)
    if pca is not None:
        query_combined = pca.transform(query_combined.reshape(1, -1))[0]
    distances = []
    for img_path, feat_dict in database.items():
        db_combined = combine_features(feat_dict, selected_keys)
        if pca is not None:
            db_combined = pca.transform(db_combined.reshape(1, -1))[0]
        if similarity == "Cosine":
            d = cosine_distance(query_combined, db_combined)
        elif similarity == "Cross-bin":
            d = cross_bin_dissimilarity(query_combined, db_combined)
        elif similarity == "Jensen-Shannon":
            d = jensen_shannon_divergence(query_combined, db_combined)
        elif similarity == "Jaccard":
            sim_val = jaccard_similarity(query_combined, db_combined)
            d = 1.0 - sim_val
        else:
            d = cosine_distance(query_combined, db_combined)
        distances.append((img_path, d))
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]