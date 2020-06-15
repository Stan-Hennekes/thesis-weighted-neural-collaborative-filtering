import logging

import implicit

LOGGER = logging.getLogger(__name__)


def train_als(train_csr, n_factors, lambda_reg, n_iterations):
    """
    Trains a WMF model using Alternating Least Squares to the sparse interaction matrix train_csr.

    Args:
        train_csr (csr_matrix): A (sparse) interaction matrix of type csr where the elements
        represent the confidence of interest of a user in an item. This is a positive number if the
        user interacted with the item, 0 if not.
        n_factors (int): Dimension of latent factors in MF.
        lambda_reg (float): Regularization parameter.
        n_iterations (int): Number of iterations of ALS.

    Returns:
        Tuple[implicit.als.AlternatingLeastSquares, np.array, np.array]: Model, user factor matrix and item factor
        matrix.

    """
    # Initialize the ALS model and fit it using the sparse item-user matrix.
    als_model = implicit.als.AlternatingLeastSquares(
        factors=n_factors,
        regularization=lambda_reg,
        iterations=n_iterations,
        use_native=True,
        use_cg=False,
        calculate_training_loss=False,
    )
    # Fit the model.
    als_model.fit(train_csr, show_progress=True)
    return als_model, als_model.user_factors, als_model.item_factors
