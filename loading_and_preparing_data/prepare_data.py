import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix

from loading_and_preparing_data.filter_data import (
    select_popular_items,
    select_active_customers,
)

LOGGER = logging.getLogger(__name__)


def prepare_interaction_df(user_hub_filter, level, config):
    """
    Read raw purchase data from local .tsv file, filter it and add confidence levels.

    Args:
        user_hub_filter (string): Which filter to use to select users. Either a hub ID or 'all'.
        level (string): Which level of aggregation to use for the item. In ['article', 'l4', 'l1'].
        config (dict): Dictionary with configuration.

    Returns:
        pandas.DataFrame: DataFrame containing filtered purchase data and confidence measure.

    """

    filename = f"sets/thesis_set_anonymized.tsv"
    df_interactions = pd.read_csv(filename, sep="\t")

    # Add extra columns.
    df_interactions["sold"] = df_interactions["total_sales"] > 0
    df_interactions["confidence"] = 1 + config["alpha_confidence"] * np.log(
        1 + df_interactions["total_sales"]
    )
    if config["split_train_test_query"]:
        df_interactions["confidence_test"] = 1 + config["alpha_confidence"] * np.log(
            1 + df_interactions["total_sales_test"]
        )

    # Filter items on popularity.
    LOGGER.info("Shape of interaction df before filtering: %s", df_interactions.shape)

    filtered_df_interactions = select_active_customers(
        df_interactions,
        config["min_articles_purchased"],
        config["max_articles_purchased"],
        config["top_users_percentage"],
    )
    filtered_df_interactions = select_popular_items(
        filtered_df_interactions, config["item_popularity_threshold"]
    )
    if config["split_train_test_query"] and config["both_in_train_and_test"]:
        # Take only users and items that appear both in train and test set
        train = filtered_df_interactions[filtered_df_interactions["total_sales"] > 0]
        test = filtered_df_interactions[
            filtered_df_interactions["total_sales_test"] > 0
        ]

        train_users = list(train.groupby(["customer"]).first().index)
        test_users = list(test.groupby(["customer"]).first().index)
        users_appear_in_both = list(set(train_users).intersection(test_users))

        train_items = list(train.groupby(["article"]).first().index)
        test_items = list(test.groupby(["article"]).first().index)
        items_appear_in_both = list(set(train_items).intersection(test_items))

        filtered_df_interactions = filtered_df_interactions[
            (filtered_df_interactions["customer"].isin(users_appear_in_both))
            & (filtered_df_interactions["article"].isin(items_appear_in_both))
        ]

    LOGGER.info(
        "Shape of interaction df after filtering: %s", filtered_df_interactions.shape
    )

    if config["cut_most_pop_items"]:
        n = 10
        item_pops = (
            filtered_df_interactions[["article", "total_sales"]]
            .groupby(["article"])
            .sum()
        )
        sorted_item_pops = item_pops.sort_values(by="total_sales", ascending=False)
        most_pop_n = list(sorted_item_pops[0:n].index)
        filtered_df_interactions = filtered_df_interactions[
            ~filtered_df_interactions["article"].isin(most_pop_n)
        ]
    return df_interactions, filtered_df_interactions


def encrypt_users_and_items(df_interactions):
    """
    Encrypt user and item names to integers and create a lookup frame so we can get the article names back in readable
    form later.

    Args:
        df_interactions (DataFrame): Table containing purchase data as returned by prepare_interaction_df.

    Returns:
        DataFrame (3x): Encrypted DataFrame, DataFrame to look up original item names and DataFrame to look up original
        user names.

    """

    # Convert names into numerical IDs.
    df_interactions = df_interactions.assign(
        user_id=df_interactions["customer"].astype("category").cat.codes
    )
    df_interactions = df_interactions.assign(
        article_id=df_interactions["article"].astype("category").cat.codes
    )

    # Create a lookup frame so we can get the article names back in readable form later.
    item_lookup = df_interactions[
        [
            "article_id",
            "article",
            "art_p_art_name",
            "art_p_cat_lev_4",
            "art_p_brand_tier",
            "art_p_brand",
        ]
    ].drop_duplicates()
    user_lookup = df_interactions[
        ["user_id", "customer", "cust_household_type"]
    ].drop_duplicates()

    encrypted_df_interactions = df_interactions.drop(["customer", "article"], axis=1)

    return encrypted_df_interactions, item_lookup, user_lookup


def create_sparse_interaction_matrix(
    encrypted_df_train, encrypted_df_test, encrypted_df_interactions, use_confidence
):
    """
    Create sparse interaction matrix from DataFrame with interactions.

    Args:
        encrypted_df_interactions (DataFrame): encrypted DataFrame as returned by encrypt_users_and_items (to make
        sure all user-item combinations are generated).
        encrypted_df_train (DataFrame): encrypted DataFrame of train interactions.
        encrypted_df_test (DataFrame): encrypted DataFrame  of test interactions.
        use_confidence (Bool): Indicator specifying whether to use confidence levels or pure purchase data.

    Output:
        sparse_item_user_matrix: Sparse matrix containing pure interaction (interacted Yes/No) or confidence metric
        based on quantity ordered.

    """

    LOGGER.info("shape of full set: %s", encrypted_df_interactions.shape)
    LOGGER.info("shape of train set: %s", encrypted_df_train.shape)
    LOGGER.info("shape of test set: %s", encrypted_df_test.shape)

    # Create lists of all unique users and items, sales and confidences.
    users = np.sort(encrypted_df_interactions["user_id"].unique())
    items = np.sort(encrypted_df_interactions["article_id"].unique())

    # Get the rows and columns for our new matrix (full).
    item_locs = encrypted_df_interactions["article_id"].astype(int)
    user_locs = encrypted_df_interactions["user_id"].astype(int)

    # Get the rows and columns for our new matrix (train).
    item_locs_train = encrypted_df_train["article_id"].astype(int)
    user_locs_train = encrypted_df_train["user_id"].astype(int)

    # Get the rows and columns for our new matrix (test).
    item_locs_test = encrypted_df_test["article_id"].astype(int)
    user_locs_test = encrypted_df_test["user_id"].astype(int)

    # Construct a sparse matrix for our users and items.
    if use_confidence:
        sparse_item_user_matrix = csr_matrix(
            (encrypted_df_interactions["confidence"], (item_locs, user_locs)),
            shape=(len(items), len(users)),
        )
        sparse_item_user_matrix_train = csr_matrix(
            (encrypted_df_train["confidence"], (item_locs_train, user_locs_train)),
            shape=(len(items), len(users)),
        )
        sparse_item_user_matrix_test = csr_matrix(
            (encrypted_df_test["confidence"], (item_locs_test, user_locs_test)),
            shape=(len(items), len(users)),
        )
    else:
        # NOTE: used to not work in combination with 'timesplit_in_query'.
        sparse_item_user_matrix = csr_matrix(
            (encrypted_df_interactions["total_sales"], (item_locs, user_locs)),
            shape=(len(items), len(users)),
        )
        sparse_item_user_matrix_train = csr_matrix(
            (encrypted_df_train["total_sales"], (item_locs_train, user_locs_train)),
            shape=(len(items), len(users)),
        )
        sparse_item_user_matrix_test = csr_matrix(
            (encrypted_df_test["total_sales"], (item_locs_test, user_locs_test)),
            shape=(len(items), len(users)),
        )

    return (
        sparse_item_user_matrix,
        sparse_item_user_matrix_train,
        sparse_item_user_matrix_test,
    )


def transform_interactions(Train, alpha, epsilon=1, normalize=False, show_plot=True):
    """
    Take sparse interaction matrix and calculates confidence levels from the interactions. Essentially this is a log
    transform of all positive elements in the sparse matrix.

    Args:
        Train (csr_matrix): Sparse interaction matrix with train data
        alpha (float): Any real number to put more or less weight on the number of purchases.
        epsilon (float): Tuning parameter for confidence levels.
        normalize (bool): Boolean specifiying whether or not to normalize the confidence levels.
        show_plot (bool): Boolean specifiying whether or not to show a histogram with weights.

    Returns:
        csr_matrix: Sparse interaction matrix filled with confidence levels.

    """

    item_pops = Train.sum(axis=1).A1
    item_counts = np.array(np.diff(Train.indptr))
    av_amount = item_pops / item_counts
    av_amount[np.isnan(av_amount)] = 1  # if item not present in train set

    weighed_train = Train.tocsr(copy=True)
    weighed_train = sp.csr_matrix((weighed_train.T / av_amount).T.A)

    if alpha > -1:
        weighed_train.data[:] = 1 + alpha * np.log(1 + weighed_train.data[:] / epsilon)

    if normalize:
        weighed_train_norm = pp.normalize(weighed_train.toarray())
        weighed_train = sp.csr_matrix(weighed_train_norm)

    if show_plot:
        plt.hist(list(weighed_train.data), range=(0.01, 20), bins=100)
        plt.title("Histogram of weighed train data: r_{ui}")
        plt.show()

    return weighed_train
