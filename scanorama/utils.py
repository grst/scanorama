from fbpca import pca
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [10.0, 9.0]
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys

np.random.seed(0)

def dispersion(X):
    mean = X.mean(0)
    dispersion = np.zeros(mean.shape)
    nonzero_idx = np.nonzero(mean > 1e-10)[1]
    X_nonzero = X[:, nonzero_idx]
    nonzero_mean = X_nonzero.mean(0)
    nonzero_var = (X_nonzero.multiply(X_nonzero)).mean(0)
    temp = (nonzero_var / nonzero_mean)
    dispersion[mean > 1e-10] = temp.A1
    dispersion[mean <= 1e-10] = float('-inf')
    return dispersion
        
def reduce_dimensionality(X, dim_red_k=100):
    k = min((dim_red_k, X.shape[0], X.shape[1]))
    U, s, Vt = pca(X, k=k) # Automatically centers.
    return U[:, range(k)] * s[range(k)]

def visualize_cluster(coords, cluster, cluster_labels,
                      cluster_name=None, size=1, viz_prefix='vc',
                      image_suffix='.svg'):
    if not cluster_name:
        cluster_name = cluster
    labels = [ 1 if c_i == cluster else 0
               for c_i in cluster_labels ]
    c_idx = [ i for i in range(len(labels)) if labels[i] == 1 ]
    nc_idx = [ i for i in range(len(labels)) if labels[i] == 0 ]
    colors = np.array([ '#cccccc', '#377eb8' ])
    image_fname = '{}_cluster{}{}'.format(
        viz_prefix, cluster, image_suffix
    )
    plt.figure()
    plt.scatter(coords[nc_idx, 0], coords[nc_idx, 1],
                c=colors[0], s=size)
    plt.scatter(coords[c_idx, 0], coords[c_idx, 1],
                c=colors[1], s=size)
    plt.title(str(cluster_name))
    plt.savefig(image_fname, dpi=500)
        
def visualize_expr(X, coords, genes, viz_gene, image_suffix='.svg',
                   new_fig=True, size=1, viz_prefix='ve'):
    genes = [ gene.upper() for gene in genes ]
    viz_gene = viz_gene.upper()
    
    if not viz_gene.upper() in genes:
        sys.stderr.write('Warning: Could not find gene {}\n'.format(viz_gene))
        return
    
    image_fname = '{}_{}{}'.format(
        viz_prefix, viz_gene, image_suffix
    )

    # Color based on percentiles.
    x_gene = X[:, list(genes).index(viz_gene)].toarray()
    colors = np.zeros(x_gene.shape)
    n_tiles = 100
    prev_percentile = min(x_gene)
    for i in range(n_tiles):
        q = (i+1) / float(n_tiles) * 100.
        percentile = np.percentile(x_gene, q)
        idx = np.logical_and(prev_percentile <= x_gene,
                             x_gene <= percentile)
        colors[idx] = i
        prev_percentile = percentile

    if new_fig:
        plt.figure()
        plt.title(viz_gene)
    plt.scatter(coords[:, 0], coords[:, 1],
                c=colors, cmap=cm.get_cmap('Reds'), s=size)
    plt.savefig(image_fname, dpi=500)
