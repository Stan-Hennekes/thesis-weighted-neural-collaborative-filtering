import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

from evaluating.evaluation_functions import (
    calculate_ndcg_k,
    calculate_ndcg_k_new,
    calculate_hit_ratio_k,
    calculate_hit_ratio_k_new,
)

LOGGER = logging.getLogger(__name__)


def evaluate_item_pop(
    sparse_item_user_matrix_train, sparse_item_user_matrix_test, config
):
    """
    Evaluate model where Bhat is filled based on popularity of items.

    Args:
        sparse_item_user_matrix_train: (csr_matrix) Sparse interactions matrix to "train" on.
        sparse_item_user_matrix_test: (csr_matrix) Sparse interactions matrix to test on.
        config: (dict) Dictionary with model configuration.

    Returns:
        float (4x): evaluation metrics; NDCG and HR both normal and for new items only.

    """
    LOGGER.info("Start comparing to ItemPop")
    sparse_user_item_matrix_train = sparse_item_user_matrix_train.T
    # Calculate NDCG.
    pop_list = np.array(sparse_user_item_matrix_train.sum(axis=0))
    popscores = np.ones(sparse_user_item_matrix_train.shape) * pop_list
    ndcg_pop = calculate_ndcg_k(
        popscores, sparse_item_user_matrix_test.T, k=config["k"]
    )
    LOGGER.info("NDCG at k over test users using ItemPop: %s", ndcg_pop)
    ndcg_pop_new, ndcg_pop_new_per_user = calculate_ndcg_k_new(
        popscores,
        sparse_item_user_matrix_test.T,
        sparse_item_user_matrix_train.T,
        k=config["k"],
    )
    LOGGER.info(
        "NDCG at k of new items over test users using ItemPop: %s", ndcg_pop_new
    )
    hr_pop = calculate_hit_ratio_k(
        popscores, sparse_item_user_matrix_test.T, k=config["k"]
    )
    LOGGER.info("HR at k of over test users using ItemPop: %s", hr_pop)
    hr_pop_new = calculate_hit_ratio_k_new(
        popscores,
        sparse_item_user_matrix_test.T,
        sparse_item_user_matrix_train.T,
        k=config["k"],
    )
    LOGGER.info("HR at k of new items over test users using ItemPop: %s", hr_pop_new)
    return ndcg_pop, ndcg_pop_new, hr_pop, hr_pop_new


def sort_on_popularity(sparse_item_user_matrix):
    """
    Take a sparse interaction matrix and returns a sorted DataFrame containing the popularity
    of each item and a list of the items sorted on popularity.

    Args:
        sparse_item_user_matrix (csr_matrix): Sparse interaction matrix of type csr, typically the same train
            set you use for your models.

    Returns:
        pandas.DataFrame: Dataframe containing item IDs and their popularity.
        list: List of item IDs sorted by popularity (descending).

    """
    popularity = sparse_item_user_matrix.sum(axis=1)
    popularity_df = pd.DataFrame(popularity, columns=["Popularity"])
    sorted_popularity_df = popularity_df.sort_values(
        by="Popularity", ascending=False, na_position="last"
    )
    popularity_list = list(sorted_popularity_df.index)
    return sorted_popularity_df, popularity_list


def create_poplists(sparse_item_user_matrix, k=10):
    """
    Take a sparse interaction matrix of type csr and return an array of the k most popular products
    for each user in the interaction matrix. This makes it possible to evaluate the popularity baseline
    using the evaluation metrics for MF models.

    Args:
        sparse_item_user_matrix (csr_matrix): Sparse interaction matrix of type csr, typically the same train
            set you use for your models.
        k (optional, integer): The length of list to be recommended to each user.

    Returns:
        numpy.array: Array that contains the recommended list for each user based on Itempop. That is,
            the item_ids of the k most popular products. Note that this array should contain the same
            row for each user.

    """
    sorted_pop_df, pop_list = sort_on_popularity(sparse_item_user_matrix)
    popularity_list = pop_list[0:k]
    n_users = sparse_item_user_matrix.shape[1]
    popularity_lists = np.array([popularity_list] * n_users)
    return popularity_lists


def create_poplists_new(c_ui, k=10):
    """
    Take a sparse interaction matrix of type csr and returns an array of the k most popular products
    for each user in the interaction matrix without already bought ones.

    This makes it possible to evaluate the popularity baseline using the evaluation metrics for MF models.

    Args:
        c_ui (csr_matrix): Sparse interaction matrix of type csr, typically the same train
            set you use for your models.
        k (optional, integer): The length of list to be recommended to each user.

    Returns:
        numpy.array: Array that contains the recommended list for each user based on Itempop. That is,
            the item_ids of the k most popular products. Note that this array should contain the same
            row for each user.

    """
    c_iu = c_ui.T
    sorted_pop_df, pop_list = sort_on_popularity(c_iu)
    n_users = c_iu.shape[1]
    poplists = np.ones((n_users, k)) - 1
    for u in range(n_users):
        sorted_pop_u = sorted_pop_df.T.copy()

        # Filter out already bought items from reclists.
        c_u = np.array(sp.csr_matrix.todense(c_ui[u, :])).flatten()
        already_bought = np.where(c_u > 0)[0]

        # Set these scores very low.
        sorted_pop_u[already_bought] = -100

        # Sort predictions per user.
        predictions_u_df = pd.DataFrame(sorted_pop_u).T
        predictions_u_selected = predictions_u_df[predictions_u_df["Popularity"] >= 0]
        if len(predictions_u_selected) >= k:
            topK = list(predictions_u_selected[:k].index)
            poplists[u, :] = topK
        else:
            poplists[u, 0 : len(predictions_u_selected)] = list(
                predictions_u_selected.index
            )
    return poplists
