import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import umap
from itertools import combinations
import heapq
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering, linkage, dendrogram


def qMatrix(df: np.ndarray):
    """Optimized version of computing the Q-matrix.
    The expensive stuff is done at most (n ( n + 1 ) / 2) times.
    """
    assert( isinstance(df, np.ndarray) )
    assert( len(df.shape) == 2 )
    nFeatures = df.shape[1]
    m = np.zeros(shape=(nFeatures, nFeatures))

    setSizes = [ len(set(df[:, j])) for j in range(nFeatures) ]

    def q(i,j):
      if i == j or setSizes[i] < 1 or setSizes[j] < 1:
        return 0.0, 0.0

      nPairs = len(set(zip(df[:,i], df[:,j])))
      q1, q2 = 0.0, 0.0
      if setSizes[j] >= 2:
        q1 = (nPairs - setSizes[i]) / (setSizes[i] * (setSizes[j] - 1))
      if setSizes[i] >= 2:
        q2 = (nPairs - setSizes[j]) / (setSizes[j] * (setSizes[i] - 1))
      return q1, q2

    for i in range(nFeatures):
      for j in range(i + 1):
        q1, q2 = q(i, j)
        m[i, j] = q1
        m[j, i] = q2

    return m





def manual_linkage(q_matrix):
    """ manual calculation of single-linkage matrix using a min-heap.
    """
    # Calculate the distance matrix as the average of Q_matrix and its transpose
    dist_matrix = (q_matrix + q_matrix.T) / 2
    n = len(dist_matrix)
    Z = np.zeros((n - 1, 4))
    
    # Store the current cluster ID for each original data point
    current_cluster_id = list(range(n))
    
    # Priority queue stores tuples: (distance, original_idx1, original_idx2)
    pq = []
    
    # Initialize the heap with all pairwise distances between original indices
    for i, j in combinations(range(n), 2):
        heapq.heappush(pq, (dist_matrix[i, j], i, j))
    
    next_cid = n
    
    for k in range(n - 1):
        while True:
            d, i, j = heapq.heappop(pq)
            
            # Find the representative cluster for each original index
            rep_i = current_cluster_id[i]
            rep_j = current_cluster_id[j]

            # If the clusters have already been merged, skip
            if rep_i == rep_j:
                continue
            
            # The clusters are a valid pair to merge
            break
        
        # Record the merge in the linkage matrix
        Z[k, 0] = min(rep_i, rep_j)
        Z[k, 1] = max(rep_i, rep_j)
        Z[k, 2] = d
        Z[k, 3] = k + 2 

        # Update the cluster IDs of the merged original data points
        for l in range(n):
            if current_cluster_id[l] == rep_i or current_cluster_id[l] == rep_j:
                current_cluster_id[l] = next_cid

        # Prepare for the next merge
        next_cid += 1

    return Z



def plot_manual_dendrogram(q_matrix, labels, line_color='#4285F4'):
    """
    Plots a dendrogram using a manually created linkage matrix.
    """
    # Calculate the distance matrix as the average of Q_matrix and its transpose
    dist_matrix = (q_matrix + q_matrix.T) / 2
    n = len(labels)
    Z = manual_linkage(dist_matrix)
    fig, ax = plt.subplots(figsize=(max(3.5, n * 0.25), 4))
    
    Z_opt = optimal_leaf_ordering(Z, squareform(dist_matrix, checks=False))
    order = leaves_list(Z_opt)
    
    coords = {i: (p, 0) for p, i in enumerate(order)}
    
    for k, (c1, c2, d, _) in enumerate(Z_opt, start=n):
        x1, h1 = coords[c1]
        x2, h2 = coords[c2]
        
        ax.plot([x1, x1], [h1, d], c=line_color)
        ax.plot([x2, x2], [h2, d], c=line_color)
        ax.plot([x1, x2], [d, d], c=line_color)
        
        coords[k] = ((x1 + x2) / 2, d)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=10)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, max(Z[:, 2]) * 1.05)
    ax.set_ylabel("Distance")
    plt.tight_layout()
    #plt.savefig("Sim_4_dendrogram.png", dpi=300, bbox_inches='tight')
    plt.show()


    
def visualize_dendrogram(
    data_df: pd.DataFrame,
    q_matrix: np.ndarray,
    plot_title: str = "Dendrogram of Dependencies",
    save_path: str = None
):
    """
    Performs hierarchical clustering on a distance matrix derived from Q_matrix
    and visualizes the result as a dendrogram.

    The distance matrix is calculated as the average of Q_matrix and its transpose.
    The dendrogram is plotted using 'single' linkage method.

    Args:
        data_df (pd.DataFrame): The original DataFrame from which feature names are
                                 retrieved to be used as labels for the dendrogram.
        q_matrix (np.ndarray): The base matrix from which the distance matrix will be derived.
        plot_title (str, optional): The title of the plot.
                                    Defaults to "Dendrogram of Dependencies".
        save_path (str, optional): The file path to save the plot. If None, the plot is shown.
    """
    print("Calculating distance matrix and generating dendrogram...")
    
    # Calculate the distance matrix as the average of Q_matrix and its transpose
    distance_matrix = (q_matrix + q_matrix.T) / 2

    # Get feature names from the data DataFrame
    labels = data_df.T.index

    # Convert the square distance matrix to a condensed form (vector)
    D_avg = squareform(distance_matrix)

    # Perform hierarchical clustering using the 'single' linkage method
    Z_avg = linkage(D_avg, method='single')

    # Plot the dendrogram
    plt.figure(figsize=(4, 5))
    dendrogram(Z_avg, labels=labels, leaf_rotation=90, color_threshold=0)
    plt.title(plot_title)
    plt.ylabel("Distance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=700, bbox_inches='tight')
        print(f"Dendrogram saved to: {save_path}")
    else:
        plt.show()

    
    
def visualize_umap_embeddings(
    data_df: pd.DataFrame,
    umap_params: dict,
    q_matrix: np.ndarray,
    workflow_type: str = 'single',
    umap_fusion_params: dict = None,
    plot_title: str = "UMAP embedding of features",
    save_path: str = None
) -> np.ndarray:
    """
    Performs UMAP dimensionality reduction and visualizes the results.

    This function supports two distinct workflows based on a single input matrix (Q_matrix):
    1. 'single': Calculates a avearge distance matrix from Q_matrix and applies UMAP directly.
    2. 'concat': Calculates two separate matrices (forward and backward) from Q_matrix,
       concatenates the embeddings, and then applies a second UMAP on the concatenated data.

    Args:
        data_df (pd.DataFrame): The original DataFrame used to retrieve feature names
                                 for plotting labels (e.g., `data.T.index`).
        umap_params (dict): A dictionary of parameters for the UMAP model.
                            e.g., {'n_neighbors': 3, 'min_dist': 0.1, 'metric': 'precomputed'}.
        q_matrix (np.ndarray): The base matrix from which other matrices will be derived.
        workflow_type (str, optional): The type of UMAP workflow to run.
                                       Must be 'single' or 'concat'. Defaults to 'single'.
        umap_fusion_params (dict, optional): Parameters for the second UMAP model in the 'concat' workflow.
                                             If not provided, a default will be used.
        plot_title (str, optional): The title of the plot. Defaults to "UMAP embedding of features".
        save_path (str, optional): The file path to save the plot. If None, the plot is shown.

    Returns:
        np.ndarray: The final 2D UMAP embedding.
    """
    feature_names = data_df.T.index
    final_embedding = None

    # --- Workflow 1: Single UMAP on one matrix ---
    if workflow_type == 'single':
        print("Running UMAP on a single matrix derived from Q...")
        # Calculate the distance matrix as the average of Q and its transpose
        distance_matrix = (q_matrix + q_matrix.T) / 2
        
        reducer = umap.UMAP(**umap_params)
        final_embedding = reducer.fit_transform(distance_matrix)

    # --- Workflow 2: UMAP on two matrices, then on concatenated embeddings ---
    elif workflow_type == 'concat':
        print("Running UMAP on two matrices, then on concatenated embeddings...")
        
        # Calculate forward and backward matrices based on the formulas
        q_symmetric = np.triu(q_matrix) + np.triu(q_matrix, 1).T
        q_trans_symmetric = np.triu(q_matrix.T) + np.triu(q_matrix.T, 1).T
        
        # Default parameters for the second UMAP model
        if umap_fusion_params is None:
            umap_fusion_params = {'n_neighbors': 4, 'min_dist': 0.1, 'metric': 'euclidean', 'random_state': 42}

        # Apply UMAP to the two separate matrices
        umap_model = umap.UMAP(**umap_params)
        e_fwd = umap_model.fit_transform(q_symmetric)
        e_bwd = umap_model.fit_transform(q_trans_symmetric)

        # Concatenate the embeddings
        e_concat = pd.concat([pd.DataFrame(e_fwd), pd.DataFrame(e_bwd)], axis=1)

        # Apply UMAP on the concatenated embeddings
        umap_fusion = umap.UMAP(**umap_fusion_params)
        final_embedding = umap_fusion.fit_transform(e_concat)

    # --- Invalid workflow type ---
    else:
        raise ValueError("Invalid `workflow_type`. Choose 'single' or 'concat'.")

    # --- Visualization ---
    plt.figure(figsize=(6, 6))
    plt.scatter(final_embedding[:, 0], final_embedding[:, 1])

    # Label points with feature names
    for i, feat in enumerate(feature_names):
        plt.text(final_embedding[i, 0] + 0.01, final_embedding[i, 1] + 0.01, feat)

    plt.title(plot_title)
    plt.xlabel("UMAP_0")
    plt.ylabel("UMAP_1")

    if save_path:
        plt.savefig(save_path, dpi=700, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        #plt.savefig("Sim_4_umap.png", dpi=300, bbox_inches='tight')
        plt.show()

    return final_embedding

### Projection

def qMatrixToForceScalers(qMatrix):
    scalers = qMatrix - 0.5
    for n in range(scalers.shape[0]):
        scalers[n,n] = 0.0
    return scalers

def createPoints(nPoints):
    points = [[100*np.sin(float(n)), 100*np.cos(float(n))] for n in range(nPoints)]
    points = np.array(points)
    return points

def step(pts, scalers, f = 1.0, fieldSize=100.0):
    # Calculate the sum of forces pulling and pushing to each point.
    # Think of the forces that would pull to planets.
    d = (f * scalers) @ pts

    # Apply the forces to the points.
    pts = pts - d

    # Move the points to the center of the coordinate system
    c = np.sum(pts,axis=0) / pts.shape[0]
    pts = pts - c

    # Rescale so our points will fill the expected square
    m = np.max(np.abs(pts))
    pts = ((fieldSize / m) * pts)
    return pts



def projectFeatures(qMatrix, coolDown=0.4, fieldSize=100.0, epochs=20, drawSteps=False, columns=None, show=False):
    # Check input values
    assert( 0 <= coolDown and coolDown < 1 )
    assert( isinstance(qMatrix, np.ndarray) )
    assert( len(qMatrix.shape) == 2 )
    assert( qMatrix.shape[0] == qMatrix.shape[1] )
    nPoints = qMatrix.shape[0]
    
    # Generate matrix to compute the forces
    scalers = qMatrixToForceScalers(qMatrix)
    assert( isinstance(scalers, np.ndarray) )
    assert( len(scalers.shape) == 2 )
    assert( scalers.shape[0] == nPoints )
    assert( scalers.shape[1] == nPoints )
    assert( np.min(scalers) >= -0.5 )
    assert( np.max(scalers) <= 0.5 )
    assert( np.max(np.abs(scalers)) > 0.0 )

    # Initialize the points
    points = createPoints(nPoints)
    assert( isinstance(points, np.ndarray) )
    assert( len(points.shape) == 2 )
    assert( points.shape[0] == nPoints )
    assert( points.shape[1] == 2 )

    # move the points around
    f = 1.0
    for e in range(epochs):
        if drawSteps and e % 2 == 0:
            plt.scatter(points[:,0], points[:,1])
        points = step(points, scalers=scalers, f=f, fieldSize=fieldSize)
        f = f * (1.0 - coolDown)

    if drawSteps:
        plt.show()

    if show:
        plt.figure(figsize=(5, 5))
        plt.scatter(points[:,0], points[:,1])
        plt.xlim(-110, 110)
        plt.ylim(-110, 110)
        for i, txt in enumerate(columns):
            print(txt, (points[i,0], points[i,1]))
            plt.annotate(txt, (points[i,0], points[i,1]))
        #plt.savefig("Sim_4_projection.png", dpi=300, bbox_inches='tight')
        plt.show()

    return points
