import logging

import numpy as np

from evaluating.evaluation_functions import split_train_test
from general_functions.utils import (
    local_dump_train_test,
    take_sample_interaction_matrix,
)
from loading_and_preparing_data.prepare_data import (
    prepare_interaction_df,
    encrypt_users_and_items,
    create_sparse_interaction_matrix,
    transform_interactions,
)
from loading_and_preparing_data.visualize_interactions import visualize_interactions

LOGGER = logging.getLogger(__name__)


def load_and_prepare_interaction_matrices(filter_users, item_level, config):
    """

    Load the appropriate dataset based on config file and apply filters, train/test split and conversion to
    sparse matrices. Also does some basic visualisation and explorative measures.

    Args:
        filter_users (str): String in ['all', 'Utrecht'] or any short Hub name (like 'AMW' or 'UTS').
        item_level (str): String in ['l4', 'l1', 'article']. Data can be used on 'article', 'l4' or 'l1'
            level, corresponding to category level.
        config (dict): Configuration  dictionary.

    Returns:
        Tuple[csr_matrix, csr_matrix, np.array, np.array]: sparse user-item interaction matrices of train and test set
        (with weighted values) and lookup tables to map user/item IDs used to encode sparse matrices back to original
        keys.

    """

    sort_col = "first_bought"
    LOGGER.info("Begin loading data on level: %s", item_level)
    LOGGER.info("Using the following filter on users: %s", filter_users)

    # Choose on which set to work and import and prepare interaction matrix.
    df_interactions, filtered_df_interactions = prepare_interaction_df(
        user_hub_filter=filter_users, level=item_level, config=config,
    )
    # Show which part of the interactions is new in the test set.
    new_bought = np.sum(
        (filtered_df_interactions["total_sales_test"] > 0)
        & np.isnan(filtered_df_interactions["total_sales"])
    )
    still_bought = np.sum(
        (filtered_df_interactions["total_sales_test"] > 0)
        & (filtered_df_interactions["total_sales"] > 0)
    )
    stopped_buying = np.sum(
        np.isnan(filtered_df_interactions["total_sales_test"])
        & (filtered_df_interactions["total_sales"] > 0)
    )

    LOGGER.info(
        "Number of user-item interactions only happening in test set: %s", new_bought,
    )
    LOGGER.info(
        "Number of user-item interactions happening in both train and test set: %s",
        still_bought,
    )
    LOGGER.info(
        "Number of user-item interactions only happening in train set: %s",
        stopped_buying,
    )

    # Encrypt users and items.
    encrypted_df_interactions, item_lookup, user_lookup = encrypt_users_and_items(
        filtered_df_interactions
    )

    # Split in train and test.
    type = "timesplit_in_query" if config["split_train_test_query"] else "time"

    train, test = split_train_test(
        encrypted_df_interactions,
        train_percentage=config["train_percentage"],
        random_seed=config["random_seed"],
        type=type,
        sort_column=sort_col,
    )

    # Create sparse matrices.
    (
        sparse_item_user_matrix,
        sparse_item_user_matrix_train,
        sparse_item_user_matrix_test,
    ) = create_sparse_interaction_matrix(
        train,
        test,
        encrypted_df_interactions,
        config["use_confidence_in_data_preparation"],
    )

    sparsity_train, sparsity_test = (
        np.sum(mtrx > 0) / (mtrx.shape[0] * mtrx.shape[1])
        for mtrx in (sparse_item_user_matrix_train, sparse_item_user_matrix_test)
    )

    LOGGER.info(
        "Shape of train matrix (items, users): %s", sparse_item_user_matrix_train.shape,
    )
    LOGGER.info(
        "Shape of test matrix (items, users): %s", sparse_item_user_matrix_test.shape,
    )
    LOGGER.info("Sparsity of train matrix: %s", sparsity_train)
    LOGGER.info("Sparsity of test matrix: %s", sparsity_test)

    if config["visualize_data"]:
        visualize_interactions(
            sparse_item_user_matrix_train,
            sparse_item_user_matrix_test,
            df_interactions,
            filtered_df_interactions,
        )

    # Save train and test matrices locally.
    if config["dump_train_test"]:
        local_dump_train_test(
            sparse_item_user_matrix_train,
            sparse_item_user_matrix_test,
            item_lookup,
            user_lookup,
        )

    if config["take_sample"]:
        # NOTE: Overwrites names.
        (
            sparse_item_user_matrix_train,
            sparse_item_user_matrix_test,
        ) = take_sample_interaction_matrix(
            sparse_item_user_matrix_train,
            sparse_item_user_matrix_test,
            n_users_sample=100,
            n_items_sample=100,
            random_seed=config["random_seed"],
        )

    # Create positive weights in train set.
    # Note: works together with False use_confidence_in_data_preparation.
    if config["use_pos_weights"] or config["use_confidence"]:
        sparse_item_user_matrix_train = transform_interactions(
            sparse_item_user_matrix_train,
            config["alpha_confidence"],
            normalize=False,
            show_plot=config["visualize_data"],
        )

    return (
        sparse_item_user_matrix_train,
        sparse_item_user_matrix_test,
        user_lookup,
        item_lookup,
    )
