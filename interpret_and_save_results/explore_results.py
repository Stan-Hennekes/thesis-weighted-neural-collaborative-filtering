import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

LOGGER = logging.getLogger(__name__)


def explore_P_hat(P_hat, show_plots=False):
    """
    Calculate some metrics and shows histogram of predicted scores.

    Args:
        P_hat (np.array): Array with predicted user-item scores.
        show_plots (bool): Boolean indicating whether or not to show histogram.

    """

    if show_plots:
        # Histograms without restirctions.

        plt.hist(list(P_hat))  # Arguments are passed to np.histogram.
        plt.title("Histogram of predicted user interest in items")
        plt.show()

        plt.hist(list(P_hat.T))  # Arguments are passed to np.histogram.
        plt.title("Histogram of predicted item relevance for users")
        plt.show()

    n_elements = P_hat.shape[0] * P_hat.shape[1]
    fraction_above_1 = np.count_nonzero(P_hat > 1) / n_elements
    fraction_below_0 = np.count_nonzero(P_hat < 0) / n_elements

    LOGGER.info(
        "Max score = {:.2f}s, Min score = {:.4f}, Mean score = {:.4f}, Median score = {:.4f}, "
        "Fraction above 1 = {:.4f}, Fraction below 0  = {:.4f}".format(
            np.max(P_hat),
            np.min(P_hat),
            np.mean(P_hat),
            np.median(P_hat),
            fraction_above_1,
            fraction_below_0,
        )
    )


def explore_pos_weights(Cui_sparse):
    Cui = sp.csr_matrix.todense(Cui_sparse)
    plt.hist(np.array(Cui), bins=np.arange(1, 100, 1).tolist(), stacked=True)
    plt.title("Histogram of weights positive feedback (all users stacked)")
    plt.show()


def explore_neg_weights(c):
    plt.hist(np.array(c))
    plt.title("Histogram of weights negative feedback (all items)")
    plt.show()
