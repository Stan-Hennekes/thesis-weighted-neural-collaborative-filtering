import logging

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def split_train_test(
    df, train_percentage, random_seed=1, type="random", sort_column="date"
):
    """
    Split interactions DataFrame into a train and test set.

    3 types of train/test splits are supported.
    'Random', and 'time' can be used in combination with query LoadPurchaseData, 'timesplit_in_query'
    with LoadPurchaseData_splitted. Random does a  random split, time takes last orders as test set. Timesplit in query
    uses set that is already split in train and test in SQL query.

    Args:
        df (pandas.DataFrame): Pandas DataFrame with interaction data.
        train_percentage (float): Value between 0 and 1. Fraction of input data to use as train data. Only relevant
            if data is not already split in train and test in SQL query.
        random_seed (int): To enable duplicable output.
        type (str): Desribes how to split in train and test set. Value should be in
            ['random', 'time', 'timesplit_in_query'].
        sort_column (str): Optional for type 'date': on which column to order?

    Returns:
        pandas.DataFrame: Train set.
        pandas.DataFrame: Test set

    """

    LOGGER.info("Splitting train and test using: %s", type)
    if type == "random":
        # Split dataframe randomly.
        (train_df, test_df) = train_test_split(
            df, train_size=train_percentage, random_state=random_seed
        )
    elif type == "time":
        # Split dataframe in 'absolute time' version.
        df.sort_values(by=sort_column, ascending=True)
        n_train = round(len(df) * train_percentage)
        train_df = df[:n_train]
        test_df = df[n_train:]
    elif type == "timesplit_in_query":
        # Split dataframe in 'absolute time' version using presplit form query (Picnic only).
        train_df = df[df["total_sales"] > 0]
        test_df = df[df["total_sales_test"] > 0]
        # Make train and test look the same as normal df, with sales and confidence.
        train_df = train_df[
            ["user_id", "article_id", "total_sales", "confidence", "times_bought"]
        ]
        test_df = test_df[
            [
                "user_id",
                "article_id",
                "total_sales_test",
                "confidence_test",
                "times_bought_test",
            ]
        ]
        # Rename columns.
        for df in [train_df, test_df]:
            df.columns = [
                "user_id",
                "article_id",
                "total_sales",
                "confidence",
                "timesbought",
            ]

    return train_df, test_df


def evaluate_model(
    b_hat, sparse_item_user_matrix_train, sparse_item_user_matrix_test, k
):
    """
    Take a predicted relevance matrix and calculate NDCG and HR for it.

    Args:
        b_hat (array): Numpy array filled with predicted relevance for each user-item combination
            based on some model
        sparse_item_user_matrix_train (csr_matrix): Sparse interaction matrix that was used to train
            the model on.
        sparse_item_user_matrix_test (csr_matrix): Sparse interaction matrix to test the model on.
        k (int): Length of recommended list.

    Returns:
        float (4x): NDCG, HR and new versions averaged over users
        list: New NDCG for every user.

    """

    LOGGER.info("Start evaluating NDCG of model")
    ndcg = calculate_ndcg_k(b_hat, sparse_item_user_matrix_test.T, k)
    LOGGER.info("NDCG at k over test users: %s", ndcg)
    # On new items only.
    LOGGER.info("Start evaluating new NDCG of model")
    ndcg_new, ndcg_new_per_user = calculate_ndcg_k_new(
        b_hat, sparse_item_user_matrix_test.T, sparse_item_user_matrix_train.T, k
    )
    LOGGER.info("NDCG at k of new items over test users: %s", ndcg_new)
    hr = calculate_hit_ratio_k(b_hat, sparse_item_user_matrix_test.T, k)
    LOGGER.info("HR at k over test users: %s", hr)
    hr_new = calculate_hit_ratio_k_new(
        b_hat, sparse_item_user_matrix_test.T, sparse_item_user_matrix_train.T, k
    )
    LOGGER.info("HR at k of new items over test users: %s", hr_new)
    return ndcg, ndcg_new, hr, hr_new, ndcg_new_per_user


def calculate_ndcg_k(b_hat, c_ui_test, k):
    """
    Calculate NDCG of predicted relevance scores.

    Args:
        b_hat (np.array): Array filled with predicted relevance scores for user-item combinations.
        c_ui_test (sp.csr_matrix): Sparse user-item matrix with test interactions.
        k (int): Length of recommended lists.

    Returns:
        float: NDCG score averaged over all users.

    """

    n_users, n_items = c_ui_test.shape
    ndcgs = np.zeros(n_users)
    for u in range(n_users):
        # Take predicted relevance scores for this user.
        scores = b_hat[u, :]
        # True relevance is whether or not user bought the item in the test period.
        true_relevance = np.array(sp.csr_matrix.todense(c_ui_test[u, :])).flatten()
        true_relevance_binary = (true_relevance > 0).astype(float)
        # Reshape to use sklearn function.
        scores = np.reshape(scores, (1, len(scores)))
        true_relevance_binary = np.reshape(
            true_relevance_binary, (1, len(true_relevance_binary))
        )
        ndcg_u = ndcg_score(true_relevance_binary, scores, k)
        ndcgs[u] = ndcg_u
    ndcg = np.mean(ndcgs)
    return ndcg


def calculate_ndcg_k_new(b_hat, c_ui_test, c_ui_train, k):
    """
    Calculate NDCG of predicted relevance scores for items that the user has not already bought in the train set.

    Args:
        b_hat (np.array): Array filled with predicted relevance scores for user-item combinations.
        c_ui_test (sp.csr_matrix): Sparse user-item matrix with test interactions.
        c_ui_train (sp.csr_matrix): Sparse user-item matrix with train interactions.
        k (int): Length of recommended lists.

    Returns:
        float: NDCG score averaged over all users.
        list[float]: List of NDCG scores per user.

    """

    n_users, n_items = c_ui_test.shape
    ndcgs = np.zeros(n_users)
    for u in range(n_users):
        # To remove items already in train: give them a relevance score of 0.
        sales_train = np.array(sp.csr_matrix.todense(c_ui_train[u, :])).flatten()
        bought_in_train = (sales_train > 0).astype(float)
        scores = b_hat[u, :]
        scores_new = np.where(bought_in_train, 0, scores)

        # True relevance is whether or not user bought the item in the test period.
        true_relevance = np.array(sp.csr_matrix.todense(c_ui_test[u, :])).flatten()
        true_relevance_binary = (true_relevance > 0).astype(float)
        true_relevance_binary_new = np.maximum(
            0, true_relevance_binary - bought_in_train
        )
        new_items_in_test = np.sum(
            true_relevance_binary_new
        )  # Check if user started buying new items.

        if new_items_in_test > 0:
            # Reshape to use sklearn function.
            scores_new = np.reshape(scores_new, (1, len(scores)))
            true_relevance_binary_new = np.reshape(
                true_relevance_binary_new, (1, len(true_relevance_binary_new))
            )
            ndcg_u = ndcg_score(true_relevance_binary_new, scores_new, k)
            ndcgs[u] = ndcg_u
        else:
            ndcgs[u] = np.nan
    ndcg = np.nanmean(ndcgs)
    return ndcg, ndcgs


def calculate_hit_ratio_k(b_hat, c_ui_test, k):
    """
    Calculate Hit Ratio of predicted relevance scores.

    Args:
        b_hat (np.array): Array filled with predicted relevance scores for user-item combinations.
        c_ui_test (sp.csr_matrix): Sparse user-item matrix with test interactions.
        k (int): Length of recommended lists.

    Returns:
        float: HR score averaged over all users.

    """

    n_users, n_items = c_ui_test.shape
    hrs = np.zeros(n_users)
    for u in range(n_users):
        scores = b_hat[u, :]

        true_relevance = np.array(sp.csr_matrix.todense(c_ui_test[u, :])).flatten()
        true_relevance_binary = (true_relevance > 0).astype(float)
        n_test = np.sum(
            true_relevance_binary
        )  # Check that user bougth something in test set.
        if n_test == 0:
            n_test = 1
        combined_array = np.vstack((scores, true_relevance_binary))
        sorted_array = combined_array[:, combined_array[0, :].argsort()]
        hits_at_k = np.sum(sorted_array[1][-k:])
        hr_at_k = hits_at_k / n_test
        hrs[u] = hr_at_k
    hr = np.mean(hrs)
    return hr


def calculate_hit_ratio_k_new(b_hat, c_ui_test, c_ui_train, k):
    """
    Calculate HR of predicted relevance scores for items that the user has not already bought in the train set.

    Args:
        b_hat (np.array): Array filled with predicted relevance scores for user-item combinations.
        c_ui_test (sp.csr_matrix): Sparse user-item matrix with test interactions.
        c_ui_train (sp.csr_matrix): Sparse user-item matrix with train interactions.
        k (int): Length of recommended lists.

    Returns:
        float: HR score averaged over all users.

    """

    n_users, n_items = c_ui_test.shape
    hrs = np.zeros(n_users)
    for u in range(n_users):
        # To remove items already in train: give them a relevance score of 0.
        sales_train = np.array(sp.csr_matrix.todense(c_ui_train[u, :])).flatten()
        bought_in_train = (sales_train > 0).astype(float)
        scores = b_hat[u, :]
        scores_new = np.where(bought_in_train, 0, scores)

        true_relevance = np.array(sp.csr_matrix.todense(c_ui_test[u, :])).flatten()
        true_relevance_binary = (true_relevance > 0).astype(float)
        true_relevance_binary_new = np.maximum(
            0, true_relevance_binary - bought_in_train
        )
        n_test_new = np.sum(true_relevance_binary_new)
        if n_test_new == 0:
            n_test_new = 1

        combined_array = np.vstack((scores_new, true_relevance_binary_new))
        sorted_array = combined_array[:, combined_array[0, :].argsort()]
        hits_at_k = np.sum(sorted_array[1][-k:])
        hr_at_k = hits_at_k / n_test_new
        hrs[u] = hr_at_k
    hr = np.mean(hrs)
    return hr
