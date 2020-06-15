import numpy as np


# Select only interactions of items that are bought more often than threshold.
def select_popular_items(df_interactions_raw, item_popularity_threshold):
    """
    Select only items above a certain popularity threshold.

    Args:
        df_interactions_raw (pd.DataFrame): DataFrame with purchases.
        item_popularity_threshold (int): Minimal number of unique users to buy an item.

    Returns:
        pd.DataFrame: DataFrame with filtered purchases.

    """

    # Count how many unique users bought the items.
    occurrence_articles = (
        df_interactions_raw[["article", "total_sales"]].groupby("article").agg("count")
    )
    selected_articles = occurrence_articles[
        occurrence_articles["total_sales"] > item_popularity_threshold
    ].index
    df_interactions = df_interactions_raw[
        df_interactions_raw["article"].isin(list(selected_articles))
    ]
    return df_interactions


# Select only customers that bought a certain number of articles (or article categories).
def select_active_customers(
    df_interactions_raw,
    min_articles_purchased=None,
    max_articles_purchased=None,
    top_percentage=None,
):
    """
    Select only users above a certain activity threshold.

    Args:
        df_interactions_raw (pd.DataFrame): DataFrame with purchases.
        min_articles_purchased (int): Minimal number of unique items for a user to buy.
        max_articles_purchased (int): Maximal number of unique items for a user to buy (None if
        no limit).
        top_percentage (float): Top percentage of most active users to include in the model.

    Returns:
        pd.DataFrame: DataFrame with filtered purchases.

    """

    # Make sure max is never reached unintentionally.
    max_articles_purchased = max_articles_purchased or np.inf
    # Count how many unique items users bought.
    occurrence_customers = (
        df_interactions_raw[["customer", "total_sales"]]
        .groupby("customer")
        .agg("count")
    )
    if top_percentage == None:
        selected_customers = occurrence_customers[
            (occurrence_customers["total_sales"] > min_articles_purchased)
            & (occurrence_customers["total_sales"] < max_articles_purchased)
        ].index
        df_interactions = df_interactions_raw[
            df_interactions_raw["customer"].isin(list(selected_customers))
        ]
    else:
        n_selected = round(len(occurrence_customers) * top_percentage)
        selected_customers = occurrence_customers.sort_values(
            by="total_sales", ascending=False
        )[:n_selected].index
        df_interactions = df_interactions_raw[
            df_interactions_raw["customer"].isin(list(selected_customers))
        ]
    return df_interactions
