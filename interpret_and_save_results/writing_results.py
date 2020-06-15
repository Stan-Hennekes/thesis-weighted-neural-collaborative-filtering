from datetime import datetime

import pandas as pd


def write_reclists_to_file(
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
    B_hat,
    user_lookup,
    item_lookup,
    args,
    complete=False,
):
    """
    Write results of recommender_main function to a local Excel file.

    Args:
        reclists_named (array): Array filled with top recommendations (names) for each user according to the used model.
        poplists_named (array): Array filled with top recommendations (names) for each user according to ItemPop.
        reclists_keys (array): Array filled with top recommendations (keys) for each user according to the used model.
        choose_model (str): Which model was used to generate recommendations.
        filter_users (str): Which hub was used to train on.
        item_level (str): Whether article level or grouped level was used.
        results_dict (dict): Dictionary with NDCG and HR scores for model and baselines.
        config (dict): Dictionary with model configuration.
        user_factors (array): Latent user factors if model is WMF.
        item_factors (array): Latent item factors if model is WMF.
        B_hat (array): Predicted relevance scores for user-item combinations.
        user_lookup (DataFrame): Decryption of user IDs.
        item_lookup (DataFrame): Decryption of item IDs.
        args (dict): Dictionary with extra model configurations if model is NeuMF.
        complete (bool): Indicator of writing B_hat as well (quite a big array).

    """

    comments = ""
    timestring = datetime.now().strftime("_%Y-%m-%d_%H-%M")
    filename = (
        "scratchpad/Recommended_lists_Run" + timestring + "_" + comments + ".xlsx"
    )
    # Info about model
    dataset = "Picnic"

    df_info = pd.DataFrame(
        [[choose_model], [dataset], [filter_users], [item_level]],
        index=[
            "Model (or optimization method)",
            "Dataset",
            "User subset",
            "Item level",
        ],
        columns=["Value"],
    )
    setup = pd.DataFrame(config, index=[0]).T
    results = pd.DataFrame(results_dict, index=[0]).T

    if choose_model == "NeuMF":
        setup_neumf = pd.DataFrame(vars(args), index=[0]).T
    else:
        setup_neumf = pd.DataFrame([0])

    # Write to file
    with pd.ExcelWriter(filename) as writer:
        df_info.to_excel(writer, sheet_name="info")
        results.to_excel(writer, sheet_name="results")
        setup.to_excel(writer, sheet_name="setup")
        if choose_model == "NeuMF":
            setup_neumf.to_excel(writer, sheet_name="setup_neumf")
        pd.DataFrame(reclists_named).to_excel(writer, sheet_name="reclists")
        pd.DataFrame(reclists_keys).to_excel(writer, sheet_name="reclists_keys")
        pd.DataFrame(poplists_named).to_excel(writer, sheet_name="poplists")
        pd.DataFrame(user_factors).to_excel(writer, sheet_name="user_factors")
        pd.DataFrame(item_factors).to_excel(writer, sheet_name="item_factors")
        pd.DataFrame(user_lookup).to_excel(writer, sheet_name="user_lookup")
        pd.DataFrame(item_lookup).to_excel(writer, sheet_name="item_lookup")
        if complete:
            pd.DataFrame(B_hat).to_excel(writer, sheet_name="predicted relevance")
