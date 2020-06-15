import logging

import numpy as np
import scipy.sparse as sp

LOGGER = logging.getLogger(__name__)


def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by indices from the CSR sparse matrix mat.

    Args:
        mat (csr_matrix): Matrix with rows to be deleted.
        indices (list): Which rows should be deleted.

    Returns:
        csr_matrix: Matrix without the rows to be deleted.

    """
    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def remove_pop_items_from_evaluation(item_factors, Cui_train, Cui_test, n_top_items=10):
    """
    Delete the most popular items from model results, to not take them into account in evaluation.

    Args:
        item_factors (np.array): Item factors as predicted by a WMF model.
        Cui_train (csr_matrix): Sparse interaction matrix used to train model on.
        Cui_test (csr_matrix): Sparse interaction matrix used to test model on.
        n_top_items (int): Number of items to exclude from evaluation.

    Returns:
        np.array: Item factors without most popular items. Shape is smaller than original.
        csr_matrix: Train matrix without most popular items.
        csr_matrix: Test matrix without most popular items.
        np.array: Item factors with factors of most popular items set to 0. Shape is same as original.

    """

    LOGGER.info("Most popular items are cut. Amount of cut items: %s", n_top_items)
    pop = np.array(Cui_train.sum(axis=0))

    sorted_pop = pop.argsort()
    most_popular = sorted_pop[:, -n_top_items:]

    item_factors_cut = np.delete(item_factors, most_popular, axis=0)
    item_factors_set_pop_to_zero = item_factors
    item_factors_set_pop_to_zero[most_popular, :] = np.zeros(
        item_factors_set_pop_to_zero[most_popular, :].shape
    )

    Ciu_train = Cui_train.T.tocsr()
    Ciu_test = Cui_test.T.tocsr()

    Cui_train_cut = delete_rows_csr(Ciu_train, most_popular).T
    Cui_test_cut = delete_rows_csr(Ciu_test, most_popular).T

    return (
        item_factors_cut,
        Cui_train_cut.T,
        Cui_test_cut.T,
        item_factors_set_pop_to_zero,
    )
