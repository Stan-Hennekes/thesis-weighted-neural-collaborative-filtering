import logging
import multiprocessing
import pickle

import numpy as np
import scipy.sparse
from joblib import Parallel, delayed

from baseline_models.als_functions import train_als
from eals_model.eals_functions import item_pop_weights
from eals_model.eals_functions import (
    update_user_factors,
    update_item_factors,
    compute_S_q,
    calculate_loss,
)

LOGGER = logging.getLogger(__name__)


def train_and_save_eals(
    sparse_item_user_matrix_train, config, filter_users, item_level
):
    """
    Wrapper to train eALS using ALS initial factors and save the model to local file for later use.

    Args:
        sparse_item_user_matrix_train (csr_matrix): Weighted interaction matrix to train on.
        config (dict): Dictionary with model configurations.
        filter_users (string): Which hub was used to train on.
        item_level (string): Indicating whether data is on article or on l4 level.

    Returns:
        Tuple[implicit.als.AlternatingLeastSquares, np.array, np.array]: Model, user factor matrix and item factor
        matrix.

    """

    c = item_pop_weights(sparse_item_user_matrix_train.T, config["c_0"], config["eta"])

    # Do quick ALS search for better initialization.
    als_model_init, init_user_factors, init_item_factors = train_als(
        sparse_item_user_matrix_train,
        n_factors=config["n_factors"],
        lambda_reg=config["lambda_reg"],
        n_iterations=config["n_iterations"],
    )

    user_factors, item_factors, loss = train_eALS(
        sparse_item_user_matrix_train,
        n_factors=config["n_factors"],
        lambda_reg=config["lambda_reg"],
        n_iterations=5,
        c=c,
        calc_loss=False,
        user_factors_init=als_model_init.user_factors,
        item_factors_init=als_model_init.item_factors,
    )

    # Save as an implicit class.
    model = als_model_init
    model.user_factors = user_factors
    model.item_factors = item_factors

    # Save user and item matrices.
    modelname = "eALS_" + filter_users + "_" + item_level
    filename = "scratchpad/Models/" + modelname
    outfile = open(filename, "wb")
    pickle.dump(model, outfile)
    return model, user_factors, item_factors


def train_eALS(
    item_user_matrix_train,
    n_factors,
    lambda_reg,
    n_iterations,
    c,
    calc_loss=False,
    user_factors_init="None",
    item_factors_init="None",
):
    """
    Apply WMF model on interaction matrix using eALS optimization.

    Args:
        item_user_matrix_train (csr_matrix): Weighted interaction matrix to train on.
        n_factors (int): Number of factors to include.
        lambda_reg (float): Regularization parameter lambda.
        n_iterations (int): Number of iterations in ALS.
        c (ndarray): Item weights (of length N).

    Returns:
        ndarray (2x): Latent matrices for users and items.
        list: Value of loss function in each iteration.

    """

    Ciu = item_user_matrix_train
    Cui = Ciu.T.tocsr()
    Cui_sparse = scipy.sparse.csr_matrix(Cui)  # To make sure of type.

    n_items, n_users = Ciu.shape
    if (user_factors_init == "None") & (item_factors_init == "None"):
        # Initialize the variables randomly.
        user_factors = np.random.rand(n_users, n_factors).astype(np.float32) * 0.01
        item_factors = np.random.rand(n_items, n_factors).astype(np.float32) * 0.01
    else:
        user_factors = user_factors_init
        item_factors = item_factors_init

    num_cores = multiprocessing.cpu_count()

    losses = np.zeros(n_iterations)
    for iter in range(n_iterations):
        LOGGER.info("iteration: %s", str(iter))
        # Pre-compute S (user-independent part of the calculations) to speed up.
        S_q = compute_S_q(item_factors=item_factors, c=c)
        # Update user factors.
        LOGGER.info("start updating users")
        user_updates = Parallel(n_jobs=num_cores)(
            delayed(update_user_factors)(
                item_factors=item_factors,
                user_factors=user_factors,
                Cui=Cui_sparse,
                user=user,
                regularization=lambda_reg,
                c=c,
                S_q=S_q,
            )
            for user in range(n_users)
        )
        make_list = list(np.concatenate(user_updates, axis=0))
        # Reshape to array again.
        user_factors = np.reshape(make_list, newshape=(n_users, n_factors))
        LOGGER.info("user update done")

        # Pre-compute Sp.
        S_p = np.dot(np.transpose(user_factors), user_factors)
        # Update item factors.
        LOGGER.info("start updating items")
        item_updates = Parallel(n_jobs=num_cores)(
            delayed(update_item_factors)(
                user_factors=user_factors,
                item_factors=item_factors,
                Cui=Cui_sparse,
                item=item,
                regularization=lambda_reg,
                c=c,
                S_p=S_p,
            )
            for item in range(n_items)
        )
        make_list = list(np.concatenate(item_updates, axis=0))
        # Reshape to array again.
        item_factors = np.reshape(make_list, newshape=(n_items, n_factors))
        LOGGER.info("item update done")

        if calc_loss:
            LOGGER.info("start calculating loss")
            # Calculate loss function.
            loss = calculate_loss(
                user_factors, item_factors, Cui_sparse, regularization=lambda_reg, c=c
            )
            losses[iter] = loss
            LOGGER.info("loss: %s", str(loss))
    return user_factors, item_factors, losses
