import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def get_recommendations(model, train):
    """
    Find predicted relevance scores for all user-item combinations based on keras model.

    Args:
        model (Sequential): Trained keras model.
        train (dok_matrix): Sparse item-user matrix with train interactions.

    Returns:
        array: Array with predicted relevance for all user-item combinations.

    """

    csr_train = train.tocsr()
    B_hat = np.zeros(csr_train.shape)
    n_items = csr_train.shape[1]
    LOGGER.info("start filling b_hat")
    for u in range(csr_train.shape[0]):
        B_hat[u, :] = list(
            model.predict(
                [np.repeat([u], n_items), np.array(list(range(0, n_items + 1)))]
            )
        )

    return B_hat


def get_recommendations_info(model, train, item_lookup, user_lookup):
    """
    Find predicted relevance scores for all user-item combinations based on keras model with extra info layers.

    Args:
        model (Sequential): Trained keras model.
        train (dok_matrix): Sparse item-user matrix with train interactions.
        item_lookup (DataFrame): Table containing item IDs and extra info to use as features in model.
        user_lookup (DataFrame): Table containing user IDs and extra info to use as features in model.

    Returns:
        array: Array with predicted relevance for all user-item combinations.

    """

    csr_train = train.tocsr()
    B_hat = np.zeros(csr_train.shape)
    n_items = csr_train.shape[1]
    LOGGER.info("start filling b_hat")
    for u in range(csr_train.shape[0]):
        B_hat[u, :] = list(
            model.predict(
                [
                    np.repeat([u], n_items),
                    np.array(list(range(0, n_items + 1))),
                    np.array(list(item_lookup["art_p_brand_tier_cat"])),
                    np.array(list(item_lookup["bio"])),
                    np.repeat(
                        user_lookup["household_type_cat"][user_lookup["user_id"] == u],
                        n_items,
                    ),
                ]
            )
        )

    return B_hat


def get_recommendations_info(model, train, item_lookup, user_lookup):
    """
    Find predicted relevance scores for all user-item combinations based on keras model with extra info layers.

    Args:
        model (keras model): Trained keras model.
        train (csr_matrix): Sparse item-user matrix with train interactions.
        item_lookup (DataFrame): Table containing item IDs and extra info to use as features in model.
        user_lookup (DataFrame): Table containing user IDs and extra info to use as features in model.

    Returns:
        array: Array with predicted relevance for all user-item combinations.

    """

    csr_train = train.tocsr()
    B_hat = np.zeros(csr_train.shape)
    n_items = csr_train.shape[1]
    LOGGER.info("start filling b_hat")
    for u in range(csr_train.shape[0]):
        B_hat[u, :] = list(
            model.predict(
                [
                    np.repeat([u], n_items),
                    np.array(list(range(0, n_items + 1))),
                    np.array(list(item_lookup["art_p_brand_tier_cat"])),
                    np.array(list(item_lookup["bio"])),
                    np.repeat(
                        user_lookup["household_type_cat"][user_lookup["user_id"] == u],
                        n_items,
                    ),
                ]
            )
        )

    return B_hat
