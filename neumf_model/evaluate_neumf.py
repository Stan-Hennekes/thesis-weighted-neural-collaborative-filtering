import logging

from evaluating.evaluation_functions import (
    calculate_ndcg_k_new,
    calculate_hit_ratio_k_new,
    calculate_ndcg_k,
    calculate_hit_ratio_k,
)

from neumf_model.get_predictions_neumf import (
    get_recommendations,
    get_recommendations_info,
)

LOGGER = logging.getLogger(__name__)


def evaluate_model(model, test, train, topK, item_lookup, user_lookup, use_info):
    """
    Evaluate NeuMF model.

    Args:
        model (keras model): Trained keras model.
        test (matrix): Sparse item-user matrix with test interactions.
        train (matrix): Sparse item-user matrix with train interactions.
        topK (int): Number of items in reclist.
        item_lookup (DataFrame): Table containing item IDs and extra info to use as features in model.
        user_lookup (DataFrame): Table containing user IDs and extra info to use as features in model.
        use_info (bool): Indicator whether to use extra item and user info in model.

    Returns:
        float (4x): NDCG and HR (both normal and without items already bought in train set).
        array: U-I matrix with predicted relevances.

    """

    csr_train = train.tocsr()
    csr_test = test.tocsr()

    if use_info:
        B_hat = get_recommendations_info(model, train, item_lookup, user_lookup)
    else:
        B_hat = get_recommendations(model, train)
    LOGGER.info("Start evaluation using NDCG")
    NDCG_new, NDCGs_new = calculate_ndcg_k_new(B_hat, csr_test, csr_train, topK)
    NDCG = calculate_ndcg_k(B_hat, csr_test, topK)
    LOGGER.info("Start evaluation using HR")
    HR_new = calculate_hit_ratio_k_new(B_hat, csr_test, csr_train, topK)
    HR = calculate_hit_ratio_k(B_hat, csr_test, topK)
    return NDCG_new, HR_new, NDCG, HR, B_hat
