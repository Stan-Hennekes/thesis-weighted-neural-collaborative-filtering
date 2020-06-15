import logging
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Sets level of logging for TensorFlow

from loading_and_preparing_data.load_and_prepare_data import (
    load_and_prepare_interaction_matrices,
)
from general_functions.fit_model import fit_model
from baseline_models.popularity_baseline import evaluate_item_pop
from baseline_models.random_baseline import evaluate_random
from general_functions.utils import (
    load_model,
    map_to_0_1,
)
from interpret_and_save_results.interpret_recommended_lists import (
    make_and_name_reclists,
)
from evaluating.evaluation_functions import evaluate_model
from interpret_and_save_results.explore_results import explore_P_hat
from interpret_and_save_results.writing_results import write_reclists_to_file
from general_functions.remove_popular_items import remove_pop_items_from_evaluation

import yaml


def main(filter_users, item_level, choose_model):
    """

    Estimates and evaluates an ALS, eALS or NeuMF model and saves results to local file and/or DWH.

    Args:
        filter_users (str): String describing region to be used.
        item_level (str): String in ['l4', 'l1', 'article']. Data can be used on 'article', 'l4' or 'l1'
            level, corresponding to category level.
        choose_model (str): String in ['ALS', 'eALS', 'NeuMF']. Chooses which model to use to create
            predicted relevance scores.

    """

    with open("model_config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile)

    # Train a new model or use an existing one saved locally.
    if config["create_new_model"]:
        (
            sparse_item_user_matrix_train,
            sparse_item_user_matrix_test,
            user_lookup,
            item_lookup,
        ) = load_and_prepare_interaction_matrices(filter_users, item_level, config)

        # Train model and evaluate.
        b_hat, user_factors, item_factors, args = fit_model(
            sparse_item_user_matrix_train,
            sparse_item_user_matrix_test,
            config,
            choose_model,
            user_lookup,
            item_lookup,
            filter_users,
            item_level,
        )

    else:
        b_hat, item_lookup, user_lookup = load_model()

    if config["rescale_B"] and choose_model != "NeuMF":
        b_hat = map_to_0_1(b_hat)

    # Explore what predictions are like
    if config["explore"]:
        explore_P_hat(b_hat, show_plots=True)

    # Evaluate model
    if config["evaluate"]:
        LOGGER.info("Start evaluation")
        if config["cut_most_pop_items_in_eval"]:
            copy_sparse_item_user_matrix_train = sparse_item_user_matrix_train.copy()
            copy_sparse_item_user_matrix_test = sparse_item_user_matrix_test.copy()

            # NOTE: Overwrites names.
            (
                item_factors,
                sparse_item_user_matrix_train,
                sparse_item_user_matrix_test,
                item_factors_set_to_zero,
            ) = remove_pop_items_from_evaluation(
                item_factors,
                sparse_item_user_matrix_train.T,
                sparse_item_user_matrix_test.T,
                n_top_items=10,
            )

        results_dict = {}

        # Calculate NDCG.
        (
            results_dict["ndcg"],
            results_dict["ndcg_new"],
            results_dict["hr"],
            results_dict["hr_new"],
            ndcg_new_per_user,
        ) = evaluate_model(
            b_hat,
            sparse_item_user_matrix_train,
            sparse_item_user_matrix_test,
            config["k"],
        )

        if config["popularity_baseline"]:
            # Compare to item pop.
            (
                results_dict["ndcg_pop"],
                results_dict["ndcg_pop_new"],
                results_dict["hr_pop"],
                results_dict["hr_pop_new"],
            ) = evaluate_item_pop(
                sparse_item_user_matrix_train, sparse_item_user_matrix_test, config
            )

        if config["random_baseline"]:
            # Compare to random as well.
            (
                results_dict["ndcg_random"],
                results_dict["ndcg_random_new"],
            ) = evaluate_random(
                sparse_item_user_matrix_train, sparse_item_user_matrix_test, config
            )

        if config["int_reclists"]:
            if config["cut_most_pop_items_in_eval"]:
                item_factors = item_factors_set_to_zero
                sparse_item_user_matrix_train = copy_sparse_item_user_matrix_train
                sparse_item_user_matrix_test = copy_sparse_item_user_matrix_test

            # Interpret reclists and write to output file.
            (
                reclists_named,
                poplists_named,
                reclists_keys,
                item_lookup,
            ) = make_and_name_reclists(
                sparse_item_user_matrix_train,
                item_lookup,
                user_lookup,
                config["k"],
                item_level,
                b_hat,
            )

            # Write to file.
            LOGGER.info("Start writing to file")
            write_reclists_to_file(
                reclists_named,
                poplists_named,
                reclists_keys,
                choose_model,
                filter_users,
                item_level,
                results_dict,
                config,
                user_factors,
                item_factors,
                b_hat,
                user_lookup,
                item_lookup,
                args,
                complete=False,
            )
            LOGGER.info("Done writing to file")

    return results_dict


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)-4s: %(module)-4s :%(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER = logging.getLogger("recommender")
    results_dict = main(
        filter_users="thesis_version", item_level="article", choose_model="ALS"
    )
