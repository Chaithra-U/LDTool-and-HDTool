import pandas as pd
import numpy as np
import random
import heapq
from typing import List
from tqdm.auto import tqdm
from qFunction.qFunction import Q, Qfast


# ---[ Used in original and in fast version ]-------------------------------

class FeatureTreeNode:
    """Node for the Vantage-Point (VP) feature tree."""
    def __init__(self, feature_index, median_score=0, left=None, right=None):
        self.feature_index = feature_index  # The pivot feature for this node
        self.median_score = median_score    # The median distance (radius) for partitioning
        self.left = left                  # Subtree with features closer to the pivot
        self.right = right                # Subtree with features farther from the pivot


def build_feature_tree(qf, feature_indices=None, progress=None):
    """Recursively builds a Vantage-Point tree on the feature space."""
    if feature_indices is None:
        feature_indices = list(range(data.shape[1]))

    if len(feature_indices) <= 1:
        if feature_indices:
            if progress: progress.update(1)
            return FeatureTreeNode(feature_indices[0])
        return None

    root_feature = random.choice(feature_indices)
    remaining_features = [f for f in feature_indices if f != root_feature]


    scores = [(f, qf.calc(root_feature, f)) for f in remaining_features ]
    scores.sort(key=lambda x: x[1])

    mid_index = len(scores) // 2
    left_features = [f for f, _ in scores[:mid_index]]
    right_features = [f for f, _ in scores[mid_index:]]
    median_score = scores[mid_index][1] if mid_index < len(scores) else 0

    node = FeatureTreeNode(root_feature, median_score=median_score)
    if progress: progress.update(1)

    node.left = build_feature_tree(qf, left_features, progress)
    node.right = build_feature_tree(qf, right_features, progress)

    return node

def count_total_nodes(feature_indices):
    if not feature_indices or len(feature_indices) <= 1:
        return len(feature_indices)
    mid = len(feature_indices) // 2
    return 1 + count_total_nodes(feature_indices[:mid]) + count_total_nodes(feature_indices[mid+1:])

def build_feature_tree_with_progress(qf):
    """Entry point to build the feature tree with a progress bar."""
    assert( isinstance(qf, Q) or isinstance(qf, Qfast) )
    feature_indices = list(range(qf.nFeatures))
    total_nodes = count_total_nodes(feature_indices)
    with tqdm(total=total_nodes, desc="Building Feature Tree") as progress:
        tree = build_feature_tree(qf, feature_indices, progress)
    return tree




def knn_features(tree, query_feature, qf, k=5, ignoreResult=False):
    """Finds k nearest features using the VP-tree with efficient pruning."""
    # Use a max-heap to store the k-best neighbors found so far.
    assert( isinstance(qf, Q) or isinstance(qf, Qfast) )
    best = []

    def search(node):
        if node is None:
            return

        # 1. Calculate distance from query to the current pivot feature.
        dist_to_pivot = qf.calc(query_feature, node.feature_index)

        # The current search radius is the score of the k-th farthest neighbor found so far.
        tau = -best[0][0] if len(best) == k else float('inf')

        # 2. Update the heap of k-nearest neighbors with the current pivot.
        if node.feature_index != query_feature and dist_to_pivot < tau:
            heapq.heappush(best, (-dist_to_pivot, node.feature_index))
            if len(best) > k:
                heapq.heappop(best)

        # 3. Search the more promising sub-tree first.
        if dist_to_pivot < node.median_score:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        search(first)

        # 4. PRUNING: Search the second branch only if it can contain better neighbors.
        # This is only possible if the query ball of radius tau intersects the other partition.
        tau = -best[0][0] if len(best) == k else float('inf')
        if len(best) < k or abs(dist_to_pivot - node.median_score) <= tau:
            search(second)

    search(tree)
    if not ignoreResult:
      return sorted([(-s, f) for s, f in best])






# ---[ Used in original version only ]--------------------------------------

def compute_knn_dependency_matrix(tree, qf, k=5):
    """Computes k-nearest neighbors and dependency scores for every feature."""
    assert( isinstance(qf, Q) or isinstance(qf, Qfast) )
    n_features = qf.nFeatures
    knn_matrix = []
    score_matrix = []

    for query_feature in tqdm(range(n_features), desc="Computing kNN for all features"):
        neighbors_with_scores = knn_features(tree, query_feature=query_feature, qf=qf, k=k)

        neighbors = [f for _, f in neighbors_with_scores]
        scores = [s for s, _ in neighbors_with_scores]

        # Pad results if fewer than k neighbors were found.
        while len(neighbors) < k:
            neighbors.append(-1)
            scores.append(float('inf'))

        knn_matrix.append(neighbors)
        score_matrix.append(scores)

    return knn_matrix, score_matrix


def compute_knn_dependency_matrix(tree, data, qf, k=5):
    """Computes k-nearest neighbors and dependency scores for every feature."""
    n_features = data.shape[1]
    knn_matrix = []
    score_matrix = []

    for query_feature in tqdm(range(n_features), desc="Computing kNN for all features"):
        neighbors_with_scores = knn_features(tree, query_feature=query_feature, qf=qf, k=k)

        neighbors = [f for _, f in neighbors_with_scores]
        scores = [s for s, _ in neighbors_with_scores]

        # Pad results if fewer than k neighbors were found.
        while len(neighbors) < k:
            neighbors.append(-1)
            scores.append(float('inf'))

        knn_matrix.append(neighbors)
        score_matrix.append(scores)

    return knn_matrix, score_matrix


def create_adjacency_from_knn(
    knn_matrix: List[List[int]],
    score_matrix: List[List[float]],
    n_features: int
) -> np.ndarray:
    """
    Creates a feature dependency adjacency matrix from kNN results.
    The resulting matrix is asymmetric, as A[i,j] is the dependency score
    of feature j on i, but not necessarily vice-versa.
    """
    # 1. Initialize an empty matrix.
    adj_matrix = np.ones((n_features, n_features), dtype=np.float64)

    if not any(knn_matrix):
        return adj_matrix

    # 2. Prepare indices and scores for efficient vectorized assignment.
    row_indices = np.arange(n_features).repeat([len(k) for k in knn_matrix])
    col_indices = np.concatenate([k for k in knn_matrix if k])
    scores = np.concatenate([s for s in score_matrix if s])

    # Filter out placeholder indices (-1).
    valid_mask = col_indices != -1

    # 3. Use NumPy's advanced indexing to populate the matrix in one operation.
    adj_matrix[row_indices[valid_mask], col_indices[valid_mask]] = scores[valid_mask]

    r = np.array(range(n_features))
    adj_matrix[r,r] = 0

    return adj_matrix


def qMatrixUsingTree(data, k=5, debug=False):
  qf = Q(data)
  if debug:
    print("---[ Build Tree ]------------------------------------------------")
  tree = build_feature_tree_with_progress(qf)
  if debug:
    qf.statistics()
    print("---[ KNN ]-------------------------------------------------------")
  knn_matrix, score_matrix = compute_knn_dependency_matrix(tree, data, qf=qf, k=k)
  if debug:
    qf.statistics()
    print("---[ Matrix ]----------------------------------------------------")
  adj_matrix = create_adjacency_from_knn(knn_matrix, score_matrix, data.shape[1])
  qf.qValues = adj_matrix
  return adj_matrix, qf






# ---[ Used in fast version only ]------------------------------------------

def qMatrixUsingTreeFast(data, k=5, debug=False):
  qf = Qfast(data)
  if debug:
    print("---[ Build Tree ]------------------------------------------------")
  tree = build_feature_tree_with_progress(qf)
  if debug:
    qf.statistics()

    print("---[ KNN ]-------------------------------------------------------")

  """Computes k-nearest neighbors and dependency scores for every feature."""
  for query_feature in tqdm(range(qf.nFeatures), desc="Computing kNN for all features"):
      knn_features(tree, query_feature=query_feature, qf=qf, k=k, ignoreResult=True)

  if debug:
    qf.statistics()
    print("---[ Matrix ]----------------------------------------------------")

  """
  Creates a feature dependency adjacency matrix from the cached Q-values.
  The resulting matrix is asymmetric, as A[i,j] is the dependency score
  of feature j on i, but not necessarily vice-versa.
  """
  adj_matrix = qf.qValues
  adj_matrix[np.isnan(adj_matrix)] = 1.0
  r = np.array(range(qf.nFeatures))
  adj_matrix[r,r] = 0
  qf.qValues = adj_matrix
  return adj_matrix, qf

