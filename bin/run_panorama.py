"""Run scanorama.

Usage:
    run_scanorama.py INPUT_FILES OUTPUT_DIR

Arguments:
    INPUT_FILES file containing path to 'npz' files line by line
    OUTPUT_DIR directory in that the output files will be stored.

"""

from process import load_names, merge_datasets
from scanorama import *
from docopt import docopt


from time import time
import numpy as np

NAMESPACE = 'panorama'


def main(data_names, output_dir):
    datasets, genes_list, n_cells = load_names(data_names)

    t0 = time()
    datasets_dimred, datasets, genes = correct(
        datasets, genes_list, ds_names=data_names,
        return_dimred=True
    )
    if VERBOSE:
        print('Integrated and batch corrected panoramas in {:.3f}s'
              .format(time() - t0))

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    np.savez(datasets_dimred, output_dir + "/datasets_dimred.npz")
    np.savez(datasets, output_dir + "/datasets.npz")
    np.savez(genes, output_dir + "/genes.npz")


    embedding = visualize(datasets_dimred,
                          labels, NAMESPACE + '_ds', names,
                          multicore_tsne=False, out_dir=output_dir)

    np.savez(embedding, output_dir + '/tsne.npz')

    # Uncorrected.
    datasets, genes_list, n_cells = load_names(data_names)
    datasets, genes = merge_datasets(datasets, genes_list)
    datasets_dimred = dimensionality_reduce(datasets)

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    embedding = visualize(datasets_dimred, labels,
                          NAMESPACE + '_ds_uncorrected', names, out_dir=output_dir)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    with open(arguments['INPUT_FILES'], 'rt') as f:
        data_names = [x.strip() for x in f.readlines()]

    main(data_names, arguments['OUTPUT_DIR'])
