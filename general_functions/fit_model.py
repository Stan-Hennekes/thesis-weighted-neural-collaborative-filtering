import os

import numpy as np
import logging

from baseline_models.als_functions import train_als
from evaluating.evaluation_functions_helpers import get_recommendations
from eals_model.eals_model import train_and_save_eals
from neumf_model.neumf import NeuMF_model

LOGGER = logging.getLogger(__name__)


def fit_model(
    sparse_item_user_matrix_train,
    sparse_item_user_matrix_test,
    config,
    model_type,
    user_lookup,
    item_lookup,
    filter_users,
    item_level,
):
    """
    Fit a Collaborative Filtering model to the data. Possible models are WMF (estimated by either ALS or eALS) and
    NeuMF.

    Args:
        sparse_item_user_matrix_train (sp.csr_matrix): Train interaction matrix.
        sparse_item_user_matrix_test (sp.csr_matrix): Test interaction matrix.
        config (dict): Configuration  dictionary.
        model_type (str): String in ['ALS', 'eALS', 'NeuMF']. Chooses which model to use to create
            predicted relevance scores.
        user_lookup (pd.DataFrame): DataFrame with encrypted user IDs and names.
        item_lookup (pd.DataFrame): DataFrame with encrypted item IDs and names.
        filter_users (str): String in ['all', 'Utrecht'] or any short Hub name (like 'AMW' or 'UTS').
        item_level (str): String in ['l4', 'l1', 'article']. Data can be used on 'article', 'l4' or 'l1'
            level, corresponding to category level.

    Returns:
        np.array: Predicted relevance scores for all user-item combinations. Same size as train and test matrices.
        np.array (2x): Estimated user and item factor matrices. Only filled when choose_model is either ALS or eALS.
        dict: Used settings of Neural model. Only filled when choose_model is NeuMF.

    """
    if model_type in ("ALS", "eALS"):
        LOGGER.info("Start training " + model_type)
        if model_type == "ALS":
            model, user_factors, item_factors = train_als(
                sparse_item_user_matrix_train,
                n_factors=config["n_factors"],
                lambda_reg=config["lambda_reg"],
                n_iterations=config["n_iterations"],
            )
        else:
            model, user_factors, item_factors = train_and_save_eals(
                sparse_item_user_matrix_train, config, filter_users, item_level
            )

        B_hat = get_recommendations(
            user_factors,
            item_factors,
            c_ui=sparse_item_user_matrix_train.T,
            penalize_pop_items=False,
        )
        args = None  # Make empty (prevent error in writing).

    elif model_type == "NeuMF":
        # Make empty user and item factors (prevent error in writing).
        user_factors = np.empty((1, 1))
        item_factors = np.empty((1, 1))

        LOGGER.info("Start training NeuMF")

        B_hat, B_hat_best, args, losses, NDCGs, NDCGs_new = NeuMF_model(
            train=sparse_item_user_matrix_train.T,
            test=sparse_item_user_matrix_test.T,
            loss=config["loss"],
            use_pos_weights=config["use_pos_weights"],
            use_neg_weights=config["use_neg_weights"],
            c_0=config["c_0"],
            eta=config["eta"],
            n_negatives=3,  # var,
            lambda_emb_GMF=0,
            lambda_emb_MLP=0,
            lambda_MLP=0,
            lambda_out=0,
            user_lookup=user_lookup,
            item_lookup=item_lookup,
            use_info=config["use_info"],
        )

    return B_hat, user_factors, item_factors, args
