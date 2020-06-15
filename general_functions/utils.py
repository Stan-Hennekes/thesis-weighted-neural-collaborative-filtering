import pickle
import random
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


def local_dump_train_test(
    sparse_item_user_matrix_train,
    sparse_item_user_matrix_test,
    item_lookup,
    user_lookup,
):
    """
    Write train, test and lookups to local pickle files for later usage.

    Args:
        sparse_item_user_matrix_train (sp.csr_matrix): Train interaction matrix.
        sparse_item_user_matrix_test (sp.csr_matrix): Test interaction matrix.
        item_lookup (pd.DataFrame): Decryption of item IDs.
        user_lookup (pd.DataFrame): Decryption of user IDs.

    """

    # Save matrix.
    filename = "scratchpad/Sets/trainMatrix"
    outfile = open(filename, "wb")
    pickle.dump(sparse_item_user_matrix_train, outfile)
    outfile.close

    filename = "scratchpad/Sets/testMatrix"
    outfile = open(filename, "wb")
    pickle.dump(sparse_item_user_matrix_test, outfile)
    outfile.close

    filename3 = "scratchpad/Sets/itemLookup"
    outfile = open(filename3, "wb")
    pickle.dump(item_lookup, outfile)
    outfile.close

    filename = "scratchpad/Sets/userLookup"
    outfile = open(filename, "wb")
    pickle.dump(user_lookup, outfile)
    outfile.close


def take_sample_interaction_matrix(
    sparse_item_user_matrix_train,
    sparse_item_user_matrix_test,
    n_users_sample,
    n_items_sample,
    random_seed=1,
):
    """
    Take random sample of users and items and creates train and test interaction matrices with
    only the sampled user-item combination.

    Args:
        sparse_item_user_matrix_train (sp.csr_matrix): Train interaction matrix.
        sparse_item_user_matrix_test (sp.csr_matrix): Test interaction matrix.
        n_users_sample: (int) Number of users to sample.
        n_items_sample: (int) Number of items to sample.
        random_seed: (int) Random seed.

    Returns:
        sp.csr_matrix (2x): Sampled train and test matrices.

    """
    N, M = sparse_item_user_matrix_train.shape
    LOGGER.info("n_users_sample: %s", n_users_sample)
    LOGGER.info("n_items_sample: %s", n_items_sample)

    # Take random subset of users and items from train and test set.
    random.seed(random_seed)
    random_user_sample = random.sample(range(M), n_users_sample)
    random_item_sample = random.sample(range(N), n_items_sample)
    Cui_train_sample = sparse_item_user_matrix_train.T[random_user_sample, :][
        :, random_item_sample
    ]
    Cui_test_sample = sparse_item_user_matrix_test.T[random_user_sample, :][
        :, random_item_sample
    ]

    return Cui_train_sample.T, Cui_test_sample.T


def load_model():
    """
    Load WMF model and user and item decryption from local file.
    """

    # Load model.
    filename1 = "scratchpad/Models/eALS_DVT_article"  # Or some other model name
    infile1 = open(filename1, "rb")
    model = pickle.load(infile1)
    infile1.close()
    # Load item and user lookups.
    filename2 = "scratchpad/Sets/itemLookup"
    infile2 = open(filename2, "rb")
    item_lookup = pickle.load(infile2)
    infile2.close()
    filename3 = "scratchpad/Sets/userLookup"
    infile3 = open(filename3, "rb")
    user_lookup = pickle.load(infile3)
    infile3.close()

    return model, item_lookup, user_lookup


def scale(x, out_range=(-1, 1)):
    """
    Scales list or array of values to specified domain.

    Args:
        x (list[float]): Values that need to be rescaled.
        out_range (tuple): Lower and upper boundary of domain te rescale to.

    Returns:
        list[float]: Rescaled values.

    """

    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def map_to_0_1(B_hat):
    """
    Take array with predicted relevance scores and scale values to [0,1] range for each user.

    Args:
        B_hat (np.array): Predicited relevance scores from WMF model.

    Returns:
        np.array Rescaled predicted relevance scores.

    """

    B_hat_scaled = np.zeros(B_hat.shape)
    for u in range(B_hat.shape[0]):
        predicted_user = B_hat[u, :]
        scaled = scale(predicted_user, out_range=(0, 1))
        B_hat_scaled[u, :] = scaled
    return B_hat_scaled
