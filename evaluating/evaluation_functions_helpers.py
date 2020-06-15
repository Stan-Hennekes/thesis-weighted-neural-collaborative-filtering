import numpy as np
import pandas as pd
import scipy.sparse as sp


def get_recommendations(
    user_factors, item_factors, c_ui, penalize_pop_items=False, fraction=0.5
):
    """
    Generate predicted relevance scores from latent factors.

    Args:
        user_factors (np.array): Array with user factors.
        item_factors (np.array): Array with item factors.
        c_ui (csr_matrix): Sparse interaction matrix to base popularity on.
        penalize_pop_items (bool): Boolean indicating whether or not to put a penalty on popular items.
        fraction (float): Parameter to make penalty more or less strict.

    Returns:
        np.array: Array filled with predicted user-item relevance scores (also called B_hat).

    """

    if penalize_pop_items:
        n_users = user_factors.shape[0]
        n_items = item_factors.shape[0]
        item_pops = np.array(c_ui.sum(axis=0))
        item_biases = fraction * item_pops / np.max(item_pops)
        predictions = np.dot(
            user_factors, np.transpose(item_factors)
        ) - item_biases * np.ones((n_users, n_items))
    else:
        predictions = np.dot(user_factors, np.transpose(item_factors))
    return predictions


def make_recommended_list_all_new(Cui_train, B_hat, item_lookup, k=10):
    """
    Take predicted relevance scores and  outputs list of top K recommendations for all users without already bought
    ones.

    Args:
        Cui_train (csr_matrix): Sparse matrix with train interactions.
        B_hat (np.array): Array with predicted relevance scores.
        item_lookup (pd.DataFrame): DataFrame containing decryption of item IDs.
        k (optional, integer): Length of list to be recommended to each user.

    Returns:
        numpy.array: Array that contains the recommended list for each user based on the predicted relevance of items.

    """

    n_users = Cui_train.shape[0]
    reclists = np.zeros((n_users, k))

    for u in range(0, n_users):
        predictions_u = B_hat[u, :]

        # Filter out already bought items from reclists.
        C_u = np.array(sp.csr_matrix.todense(Cui_train[u, :])).flatten()
        already_bought = np.where(C_u > 0)[0]

        # Set these prediction very low.
        predictions_u[already_bought] = -100

        # Sort predictions per user.
        predictions_u_df = pd.DataFrame(predictions_u)
        predictions_u_sorted = predictions_u_df.sort_values(by=[0], ascending=False)
        top_k = list(predictions_u_sorted[:k].index)
        reclists[u, :] = top_k
    return reclists
