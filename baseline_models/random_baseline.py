import logging

import numpy as np

from evaluating.evaluation_functions import (
    calculate_ndcg_k,
    calculate_ndcg_k_new,
)

LOGGER = logging.getLogger(__name__)


def evaluate_random(
    sparse_item_user_matrix_train, sparse_item_user_matrix_test, config
):
    """
    Evaluate model where Bhat is filled with random relevance scores.

    Args:
        sparse_item_user_matrix_train (csr_matrix): Sparse interactions matrix to "train" on.
        sparse_item_user_matrix_test (csr_matrix): Sparse interactions matrix to test on.
        config (dict): Dictionary with model configuration.

    Returns:
        float (x2): Evaluation metrics; NDCG both normal and for new items only.

    """
    LOGGER.info("Start comparing to random")
    b_hat_random = np.random.rand(
        sparse_item_user_matrix_train.shape[1], sparse_item_user_matrix_train.shape[0]
    ).astype(np.float32)
    ndcg_random = calculate_ndcg_k(
        b_hat_random, sparse_item_user_matrix_test.T, k=config["k"]
    )
    LOGGER.info("NDCG at k of random: %s", ndcg_random)
    ndcg_random_new, ndcg_random_new_per_user = calculate_ndcg_k_new(
        b_hat_random,
        sparse_item_user_matrix_test.T,
        sparse_item_user_matrix_train.T,
        k=config["k"],
    )
    LOGGER.info("NDCG at k of new items of random: %s", ndcg_random_new)
    return ndcg_random, ndcg_random_new
