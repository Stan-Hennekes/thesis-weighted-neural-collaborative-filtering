import numpy as np
import keras.backend as kb
import os
import pandas as pd


def get_train_instances(
    train, n_items, n_neg, c, use_pos_weights=False, use_neg_weights=False
):
    """
    Take train matrix with interactions and turn it into input for NeuMF.

    Args:
        train (matrix): Sparse matrix containing train purchases.
        n_items (int): Number of different items in the set.
        n_neg (int): Number of negative instances to be sampled per positive one.
        c (array): Array of length N (number of items) with item weights.
        use_pos_weights (bool): Boolean representing whether or not the values in train matrix are taken as labels,
            or just a 1 for every user-item with a purchase.
        use_neg_weights (bool): Boolean representing whether or not to sample negatives according to the weights in c.

    Returns:
        3 np.arrays with userids, itemids and labels. Positions correspond between them:
        every position gives a user, an item, and their interaction.
    """

    user, item, labels, tiers = [], [], [], []
    # Create probabilities to be sampled (only used if use_neg_weights True).
    probs = list(c / np.sum(c))
    # Loop over all user-item combinations that are in the train set.
    for (u, i) in train.keys():
        # Add positive instances.
        user.append(u)
        item.append(i)
        if use_pos_weights:
            labels.append(train[u, i])  # The value of w_ui.
        else:
            labels.append(1)  # Binary if interaction occurred or not.
        # Sample negative instances.
        for t in range(n_neg):
            if use_neg_weights:
                # Sample a random item where the probability of each item scales with c_i.
                j = int(np.where(np.random.multinomial(1, probs))[0])
                while (u, j) in train.keys():  # Check if interaction did not occur.
                    j = np.random.randint(n_items)
            else:
                # Sample a random item.
                j = np.random.randint(n_items)
                while (u, j) in train.keys():  # Check if interaction did not occur.
                    j = np.random.randint(n_items)
            user.append(u)
            item.append(j)
            labels.append(0)
    return np.array(user), np.array(item), np.array(labels)


def get_train_instances_with_info(
    train,
    n_items,
    n_neg,
    user_lookup,
    item_lookup,
    c,
    use_pos_weights=False,
    use_neg_weights=False,
):
    """
    Take train matrix with interactions, user information and item information and turn it into input for NeuMF.

    Args:
        train (matrix): Sparse matrix containing train purchases.
        n_items (int): Number of different items in the set.
        n_neg (int): Number of negative instances to be sampled per positive one.
        item_lookup (DataFrame): Table containing item IDs and extra info to use as features in model.
        user_lookup (DataFrame): Table containing user IDs and extra info to use as features in model.
        c (array): Array of length N (number of items) with item weights.
        use_pos_weights (bool): Boolean representing whether or not the values in train matrix are taken as labels,
            or just a 1 for every user-item with a purchase.
        use_neg_weights (bool): Boolean representing whether or not to sample negatives according to the weights in c.

    Returns:
        6 np.arrays with userids, itemids, labels, brand tiers, bio labels and household types. Positions correspond
        between them: every position gives a user (with its hh type), an item (with tier and bio label), and their
        interaction.
    """

    user, item, labels, tiers, bio, household_type = [], [], [], [], [], []
    # Create probabilities to be sampled (only used if use_neg_weights True).
    probs = list(c / np.sum(c))
    # Loop over all user-item combinations that are in the train set.
    for (u, i) in train.keys():
        # Add positive instances.
        user.append(u)
        household_type_cat = list(
            user_lookup["household_type_cat"][user_lookup["user_id"] == u]
        )[0]
        household_type.append(household_type_cat)

        item.append(i)
        if use_pos_weights:
            labels.append(train[u, i])  # The value of w_ui.
        else:
            labels.append(1)  # Binary if interaction occurred or not.
        tier_cat = list(
            item_lookup["art_p_brand_tier_cat"][item_lookup["article_id"] == i]
        )[0]
        tiers.append(tier_cat)
        bio_y_n = list(item_lookup["bio"][item_lookup["article_id"] == i])[0]
        bio.append(bio_y_n)

        # Sample negative instances.
        for t in range(n_neg):
            if use_neg_weights:
                # Sample a random item where the probability of each item scales with c_i.
                j = int(np.where(np.random.multinomial(1, probs))[0])
                while (u, j) in train.keys():  # Check if interaction did not occur.
                    j = np.random.randint(n_items)
            else:
                # Sample a random item.
                j = np.random.randint(n_items)
                while (u, j) in train.keys():  # Check if interaction did not occur.
                    j = np.random.randint(n_items)
            user.append(u)
            household_type.append(household_type_cat)
            item.append(j)
            labels.append(0)
            tier = list(
                item_lookup["art_p_brand_tier_cat"][item_lookup["article_id"] == j]
            )[0]
            tiers.append(tier)
            bio_y_n = list(item_lookup["bio"][item_lookup["article_id"] == j])[0]
            bio.append(bio_y_n)
    return (
        np.array(user),
        np.array(item),
        np.array(labels),
        np.array(tiers),
        np.array(bio),
        np.array(household_type),
    )


def l2_loss():
    """
    Wrapper to use custom loss in keras backend that takes positive weights into account.
    """

    def custom_loss(y_true, y_pred):
        """
        Define L2-based loss function for NeuWMF.

        Args:
            y_true (tensor): Actual relevance of user-item combination (weighed # purchases).
            y_pred (tensor): Predicted relevance of user-item combination (score in [0,1]).

        Returns:
            tensor: Value of loss function.

        """

        y_true_bin = kb.cast(kb.greater(y_true, 0.0), "float32")
        # Return weighed square loss if true is 1, if true is 0 use weight 1.
        return kb.sum(
            y_true_bin * kb.square(y_pred - y_true_bin) * y_true
            + (1.0 - y_true_bin) * kb.square(y_pred - y_true_bin) * 1.0,
            axis=-1,
        )

    return custom_loss


def save_neumf_model_results(
    modelname,
    NDCG,
    NDCG_new,
    best_iter,
    train_time,
    use_pos_weights,
    use_neg_weights,
    loss,
    use_info,
    results_path,
):
    """
    Save results to local DataFrame.
    """

    if not os.path.isfile(results_path):
        results_df = pd.DataFrame(
            columns=["modelname", "NDCG", "NDCG_new", "best_iter", "train_time"]
        )
        experiment_df = pd.DataFrame(
            [[modelname, NDCG, NDCG_new, best_iter, train_time]],
            columns=["modelname", "NDCG", "NDCG_new", "best_iter", "train_time",],
        )
        results_df = results_df.append(experiment_df, ignore_index=True)
        results_df.to_pickle(results_path)
    else:
        results_df = pd.read_pickle(results_path)
        experiment_df = pd.DataFrame(
            [[modelname, NDCG, NDCG_new, best_iter, train_time]],
            columns=["modelname", "NDCG", "NDCG_new", "best_iter", "train_time",],
        )
        results_df = results_df.append(experiment_df, ignore_index=True)
        results_df.to_pickle(results_path)
