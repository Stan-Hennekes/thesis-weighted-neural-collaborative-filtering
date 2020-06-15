# Old version, without split in pos and neg feedback
import logging

import numpy as np
import scipy.sparse as sp

LOGGER = logging.getLogger(__name__)


def update_user_factors(item_factors, user_factors, Cui, user, regularization, c, S_q):
    """
    Performs elementwise update for each factor for a specific user u as described in Eq. 12 of He (2016,
    https://arxiv.org/abs/1708.05024).

    Args:
        item_factors (ndarray): Latent matrix for users.
        user_factors (ndarray): Latent matrix for items.
        Cui (csr_matrix): Sparse matrix with user-item interactions.
        user (int): User ID.
        regularization (float): Regularization term lambda.
        c (ndarray): Item weights (based on popularity), can be calculated by item_pop_weights().
        S_q (ndarray): Pre-computed user-independent matrix to speed up the updating process.

    Returns:
        ndarray: Updated user factors.

    """

    n_items, n_factors = item_factors.shape
    user_factors = np.copy(user_factors)  # To make it possible to write to in parallel.
    # Initialize vector for new factors for this user.
    new_user_factors_u = np.copy(user_factors[user, :])
    # Pre-compute b_ui_hat (to speed up).
    B_hat = np.dot(user_factors, np.transpose(item_factors))
    # loop over factors
    for f in range(n_factors):
        # Pre-compute negative part fo this factor.
        new_user_factors_u_nom_neg = compute_negative_part_elt(
            user_factors, S_q, user, f
        )
        # Implicit loop over all items with positive feedback for this user.
        new_user_factors_u_nom, new_user_factors_u_den = vect_update_new_user_factors_u(
            user_factors, item_factors, Cui, B_hat, c, user, f
        )
        # Out of item loop: denominator.
        new_user_factors_u_den += S_q[f, f] + regularization
        # Subtract negative part of nominator.
        new_user_factors_u_nom -= new_user_factors_u_nom_neg
        # Update factor f for this user.
        new_user_factors_u[f] = new_user_factors_u_nom / new_user_factors_u_den
        # Update general user factors matrix.
        user_factors[user, :] = new_user_factors_u
        # Update b_ui_hat with new p_uf (only user update necessary).
        B_hat[user, :] = np.dot(user_factors[user, :], np.transpose(item_factors))
        LOGGER.debug("Completed factor: %s", f)
    return new_user_factors_u


# Exactly the same function as item_factor_eALS_vectorized_pnsplit, but more readable
def update_item_factors(user_factors, item_factors, Cui, item, regularization, c, S_p):
    """
    Perform elementwise update for each factor for a specific item i as described in Eq. 13 of He (2016).

    Args:
        user_factors (ndarray): Latent matrix for users.
        item_factors (ndarray): Latent matrix for items.
        Cui (csr_matrix): Sparse matrix with user-item interactions.
        item (int): Item ID.
        regularization (float): Regularization term lambda.
        c (array): Item weights (based on popularity), can be calculated by item_pop_weights().
        S_p (array): Pre-computed item-independent matrix to speed up the updating process.

    Returns:
        ndarray: Updated item factors.

   """

    n_users, n_factors = user_factors.shape

    # Initialize vector for new factors for this item.
    new_item_factors_i = np.copy(item_factors[item, :])
    # Pre-compute b_ui_hat (to speed up).
    B_hat = np.dot(user_factors, np.transpose(item_factors))
    # loop over factors
    for f in range(n_factors):
        # Pre-compute negative part fo this factor.
        new_item_factors_i_nom_neg = c[item] * compute_negative_part_elt(
            item_factors, S_p, item, f
        )
        # Implicit loop over all users with positive feedback for this item.
        new_item_factors_i_nom, new_item_factors_i_den = vect_update_new_item_factors_i(
            user_factors, item_factors, Cui, B_hat, c, item, f
        )
        # Out of loop denominator.
        new_item_factors_i_den += c[item] * S_p[f, f] + regularization
        # Subtract negative part.
        new_item_factors_i_nom -= new_item_factors_i_nom_neg
        # New q_if.
        new_item_factors_i[f] = new_item_factors_i_nom / new_item_factors_i_den
        item_factors[item, :] = new_item_factors_i
        # Update b_ui_hat with new q_if (only item update necessary).
        B_hat[:, item] = np.dot(user_factors, np.transpose(item_factors[item, :]))
        LOGGER.debug("Completed factor: %s", f)
    return new_item_factors_i


# more helper functions
def compute_negative_part_elt(factors, S, elt, f):
    """"
    Compute sum over k in denominator of Eq. 12 or Eq. 13 of He (2016): \sum_{k \neq f} q_{ik} sp_{kf}.

    Args:
        factors (ndarray): Latent matrix for either items or users.
        S (ndarray): Either the S_q (for user updates) or S_p (for item updates) matrix.
        elt (integer): Either a specific item ID or a specific user ID.
        f {integer): A specific factor.

    Returns:
        float: Value of the sum.
    """

    # Remove factor f from factor matrix.
    elt_factors_k = np.delete(factors[elt, :], f)
    # Remove factor f from S matrix.
    S_k = np.delete(S, f, axis=0)
    # Calculate the sum over k \neq f.
    new_elt_factors_nom_neg = np.dot(elt_factors_k, S_k)[f]
    return new_elt_factors_nom_neg


def vect_update_new_user_factors_u(user_factors, item_factors, Cui, B_hat, c, u, f):
    """
    Perform sums over items in postive feedback of user u from Equation 12 of He (2016).

    Idea: do elementwise multiplication of vectors, where only the element of the vectors are taken
    that have positive feedback.

    Args:
        user_factors (ndarray): Latent matrix for users.
        item_factors (ndarray): Latent matrix for items.
        Cui (csr_matrix): Sparse matrix with user-item interactions.
        B_hat (ndarray): Current prediction matrix of size (n_users, n_items).
        c (array): Item weights (based on popularity), can be calculated by item_pop_weights().
        u (int): User ID.
        f (int): A specific factor.

    Returns:
        float: Sum over all positive feedback in nominator.
        float: Sum over all negative feedback in denominator.

    """

    C_u = Cui[u, :]
    # Make dense array to be able to do vect multiplication.
    C_u = np.array(sp.csr_matrix.todense(C_u)).flatten()
    # Which ones were bought?
    bought = C_u > 0
    Bhat_u = B_hat[u, :]
    Bhat_u_f = Bhat_u - user_factors[u, f] * item_factors[:, f]
    nom_all_items = (
        C_u[bought] - (C_u[bought] - c[bought]) * Bhat_u_f[bought]
    ) * item_factors[:, f][bought]
    nom = nom_all_items.sum()

    den_all_items = (C_u[bought] - c[bought]) * (item_factors[:, f][bought]) ** 2
    den = den_all_items.sum()
    return nom, den


def vect_update_new_item_factors_i(user_factors, item_factors, Cui, B_hat, c, i, f):
    """
    Perform sums over users in positive feedback of item i from Equation 12 of He (2016).

    Args:
        user_factors (ndarray): Latent matrix for users.
        item_factors (ndarray): Latent matrix for items.
        Cui (csr_matrix): Sparse matrix with user-item interactions.
        B_hat (ndarray): Current prediction matrix of size (n_users, n_items).
        c (array): Item weights (based on popularity), can be calculated by item_pop_weights().
        u (int): Item ID.
        f (int): A specific factor.

    Returns:
        float: Sum over all positive feedback in nominator.
        float: Sum over all negative feedback in denominator.

    """

    C_i = Cui[:, i]
    # Make dense array to be able to do vect multiplication.
    C_i = np.array(sp.csr_matrix.todense(C_i)).flatten()
    # Which ones were bought?
    bought = C_i > 0
    Bhat_i = B_hat[:, i]
    Bhat_i_f = Bhat_i - user_factors[:, f] * item_factors[i, f]
    nom_all_users = (
        C_i[bought] - (C_i[bought] - c[i]) * Bhat_i_f[bought]
    ) * user_factors[:, f][bought]
    nom = nom_all_users.sum()

    den_all_users = (C_i[bought] - c[i]) * (user_factors[:, f][bought]) ** 2
    den = den_all_users.sum()
    return nom, den


def compute_S_q(item_factors, c):
    """
    Pre-compute Sq (independent over all users).

    Args:
        item_factors (ndarray): Matrix with item factors.
        c (array): Item weights (based on popularity), can be calculated by item_pop_weights().

    Returns:
        array: Array of size (n_factors, n_factors) being Sq (see paper for more explanation).

    """

    n_items, n_factors = item_factors.shape
    # Pre-compute S (to speed up).
    Sq = np.zeros((n_factors, n_factors))
    for i in range(n_items):
        Sq += c[i] * np.outer(item_factors[i, :], np.transpose(item_factors[i, :]))
    return Sq


def calculate_loss(
    user_factors, item_factors, Cui, regularization, c
):  # Note: should be quicker!
    """
    Calculate loss function from Eq. 7 of He (2016).

    Args:
        user_factors (ndarray): Latent matrix for users.
        item_factors (ndarray): Latent matrix for items.
        Cui (csr_matrix): Sparse matrix with user-item interactions.
        regularization (float): Regularization term lambda.
        c (array): Item weights (based on popularity), can be calculated by item_pop_weights().

    Returns:
        float: Loss value.

    """

    n_users, n_factors = user_factors.shape
    n_items = item_factors.shape[0]
    B_hat = np.dot(user_factors, np.transpose(item_factors))
    L = 0
    for u in range(n_users):
        for i in range(n_items):
            if Cui[u, i] > 0:
                b_ui = 1
                L += Cui[u, i] * (b_ui - B_hat[u, i]) ** 2
            if Cui[u, i] == 0:
                L += c[i] * B_hat[u, i] ** 2
    # add regularization
    L += regularization * (
        np.linalg.norm(user_factors) ** 2 + np.linalg.norm(item_factors) ** 2
    )
    return L


def item_pop_weights(Cui, c_0, eta):
    """
    Calculate c_i from Eq. 8 of He (2016).

    Args:
        Cui (csr_matrix): Sparse matrix with user-item interactions.
        c_0 (float): Parameter to be tuned.
        eta (float): Parameter to be tuned (0.5 generally gives good results).

    Returns:
        array: Popularity weights for each item in Cui.

    """

    Cui = sp.csr_matrix(Cui)
    item_pops = Cui.sum(axis=0)
    f = item_pops / item_pops.sum()
    f_eta = np.power(f, eta)
    c = c_0 * f_eta / f_eta.sum()
    return np.array(c).flatten()
