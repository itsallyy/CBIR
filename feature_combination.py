import numpy as np
from features import FEATURE_WEIGHTS


def combine_features(features_dict, selected_keys, weights=FEATURE_WEIGHTS):
    vectors = []
    for key in selected_keys:
        v = features_dict[key]
        norm = np.linalg.norm(v)
        v_norm = v / norm if norm > 0 else v
        weighted = weights.get(key, 1.0) * v_norm
        vectors.append(weighted)
    return np.hstack(vectors)


def combine_query_feedback(query_features, feedback_list, selected_keys):
    new_query = {}
    for key in selected_keys:
        vectors = [query_features[key]]
        for feedback in feedback_list:
            vectors.append(feedback[key])
        new_query[key] = np.mean(np.vstack(vectors), axis=0)
    return new_query