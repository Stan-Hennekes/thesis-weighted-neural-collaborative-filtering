import logging

import numpy as np
import pandas as pd

from baseline_models.popularity_baseline import create_poplists_new
from evaluating.evaluation_functions_helpers import make_recommended_list_all_new

LOGGER = logging.getLogger(__name__)


def make_and_name_reclists(
    sparse_item_user_matrix_train, item_lookup, user_lookup, k, item_level, B_hat
):
    """
    Creates recommended lists based on model and ItemPop, then uses lookups to find back original
    names.

    Args:
        sparse_item_user_matrix_train (csr_matrix): Sparse interaction matrix used to train the model.
        item_lookup (pd.DataFrame): DataFrame with encrypted item IDs and names.
        user_lookup (pd.DataFrame): DataFrame with encrypted user IDs and names.
        k (int): Length of a recommended list.
        item_level (string): Whether we are working on article of grouped level.
        B_hat (array): Predicted user-item relevance scores.

    Returns:
        array (3x): Array of recommend lists of length k for each user. One filled with item names according to model,
            one with item names according to ItemPop and one with article keys according to the model.
        pd.DataFrame: DataFrame with encrypted item IDs and keys, extended with article names.

    """

    LOGGER.info("Start making reclists without already bought items")
    # Make recommended lists.
    reclists = make_recommended_list_all_new(
        Cui_train=sparse_item_user_matrix_train.T,
        item_lookup=item_lookup,
        B_hat=B_hat,
        k=k,
    )
    reclists_named = interpret_reclists(reclists, item_lookup, user_lookup)
    reclists_keys = np.zeros((1, 1))

    LOGGER.info("Start making poplists without already bought items")
    # Make recommended lists based on ItemPop.
    poplists = create_poplists_new(sparse_item_user_matrix_train.T, k)
    poplists_named = interpret_reclists(poplists, item_lookup, user_lookup)

    return reclists_named, poplists_named, reclists_keys, item_lookup


def interpret_reclists(reclists, item_lookup, user_lookup):
    """
    Find back the item and customer names from encrypted results.

    Args:
        reclists (np.array): Encrypted reclist for each user.
        item_lookup (pd.DataFrame): DataFrame with encrypted item IDs and names.
        user_lookup (pd.DataFrame): DataFrame with encrypted user IDs and names.

    Returns:
        pd.DataFrame: Decrypted recommended lists: with customer and article keys from DWH instead of IDs specific for
        this run.

    """

    item_dict = item_lookup.set_index("article_id")["article"].to_dict()
    user_dict = user_lookup.set_index("user_id")["customer"].to_dict()
    reclists_df = pd.DataFrame(reclists)
    k = reclists.shape[1]
    df_reclists_names = reclists_df
    for i in range(k):
        df_reclists_names[i] = reclists_df[i].map(item_dict)
    df_reclists_names.index = reclists_df.index.map(user_dict)
    return df_reclists_names
