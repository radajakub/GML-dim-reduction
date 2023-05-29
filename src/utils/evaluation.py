from pyDRMetrics import pyDRMetrics as pyDRM

def evaluation(data, embeddings):
    drm = pyDRM.DRMetrics(data, embeddings)
    return drm.T[5], drm.C[5]

def print_evaluation(data, embeddings):
    T, C = evaluation(data, embeddings)
    print(f"Trustworthiness: {T}")
    print(f"Continuity: {C}")