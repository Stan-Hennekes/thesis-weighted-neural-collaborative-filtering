import logging

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

LOGGER = logging.getLogger(__name__)


def visualize_interaction_frequencies(
    df_interactions,
    max_shown_user=500,
    max_shown_item=2000,
    bucket_size_item=10,
    bucket_size_user=5,
    item_summary=False,
    user_summary=False,
    show_plots=True,
):
    """
    Show histograms of popularity of items and activity of users, to study long tail. Calculats some
    simple metrics like avg items per user as well.

    Args:
        df_interactions (pandas.DataFrame): DataFrame with interaction data.
        max_shown_user (int): Last user shown in histogram has this many different items.
        max_shown_item (int): Last item shown in histogram has this many users.
        bucket_size_item (int): Bucket size in histogram of item popularity.
        bucket_size_user (int): Bucket size in histogram of user activity.
        item_summary (bool): Whether or not to show some metrics of users.
        user_summary (bool): Whether or not to show some metrics of items.

    """

    df_interactions["sold_yn"] = df_interactions["total_sales"] > 0
    # LOGGER.info('shape of interaction df to be visualised: %s', df_interactions.shape)

    # Find out number of articles per user.
    data = (
        df_interactions.groupby("customer")["total_sales"]
        .count()
        .clip(upper=max_shown_user)
    )

    if show_plots:
        # Create histogram.
        trace = go.Histogram(
            x=data.values,
            name="Articles_bought",
            xbins=dict(start=0, end=max_shown_user, size=bucket_size_user),
        )
        # Create layout.
        layout = go.Layout(
            title="Histogram of activity of users",
            xaxis=dict(title="Number of different items user interacted with"),
            yaxis=dict(title="Count of users"),
            bargap=0.2,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        # Create plot.
        fig = go.Figure(data=[trace], layout=layout)
        plot(fig)

    # Find out number of customers per article.
    data = (
        df_interactions.groupby("article")["total_sales"]
        .count()
        .clip(upper=max_shown_item)
    )

    if show_plots:
        # Create trace.
        trace = go.Histogram(
            x=data.values,
            name="Number of customers that bought the article",
            xbins=dict(start=0, end=max_shown_item, size=bucket_size_item),
        )
        # Create layout.
        layout = go.Layout(
            title="Histogram of popularity of items",
            xaxis=dict(title="Number of users that interacted with the item"),
            yaxis=dict(title="Count of items"),
            bargap=0.2,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Create plot.
        fig = go.Figure(data=[trace], layout=layout)
        plot(fig)

    if item_summary:
        # Show summary of items.
        occurrence_articles = (
            df_interactions[["article", "total_sales"]].groupby("article").agg("count")
        )
        # LOGGER.info(occurrence_articles.sort_values(by=['total_sales']))
        LOGGER.info(
            "min number of users per item: %s",
            np.min(occurrence_articles["total_sales"]),
        )
        LOGGER.info(
            "max number of users per item: %s",
            np.max(occurrence_articles["total_sales"]),
        )
        LOGGER.info(
            "mean number of users per item: %s",
            np.mean(occurrence_articles["total_sales"]),
        )
        LOGGER.info(
            "median number of users per item: %s",
            np.median(occurrence_articles["total_sales"]),
        )
        LOGGER.info("number of items: %s", len(occurrence_articles))
    if user_summary:
        # Show summary of users.
        occurrence_customers = (
            df_interactions[["customer", "total_sales"]]
            .groupby("customer")
            .agg("count")
        )
        # LOGGER.info(occurrence_customers.sort_values(by=['total_sales']))
        LOGGER.info(
            "min number of items per user: %s",
            np.min(occurrence_customers["total_sales"]),
        )
        LOGGER.info(
            "max number of items per user: %s",
            np.max(occurrence_customers["total_sales"]),
        )
        LOGGER.info(
            "mean number of items per user: %s",
            np.mean(occurrence_customers["total_sales"]),
        )
        LOGGER.info(
            "median number of items per user: %s",
            np.median(occurrence_customers["total_sales"]),
        )
        LOGGER.info("number of users: %s", len(occurrence_customers))


def visualize_interactions(
    sparse_item_user_matrix_train,
    sparse_item_user_matrix_test,
    df_interactions,
    filtered_df_interactions,
):
    """
    Make a plot of which user-item combinations had an interaction.

    Args:
        sparse_item_user_matrix_train (csr_matrix): Interaction matrix in train period.
        sparse_item_user_matrix_test (csr_matrix): Interaction matrix in test period.
        df_interactions (DataFrame): DataFrame with interaction data.
        filtered_df_interactions (DataFrame): DataFrame with filtered interaction data.

    """

    plt.spy(sparse_item_user_matrix_train, markersize=0.005)
    plt.title("Visulisation of train set")
    plt.ylabel("Item IDs")
    plt.xlabel("User IDs")
    plt.show()

    plt.spy(sparse_item_user_matrix_test, markersize=0.005)
    plt.title("Visulisation of test set")
    plt.ylabel("Item IDs")
    plt.xlabel("User IDs")
    plt.show()

    LOGGER.info("Visualize data before filtering")
    visualize_interaction_frequencies(
        df_interactions, item_summary=True, user_summary=True, show_plots=False
    )
    LOGGER.info("Visualize data after filtering")
    visualize_interaction_frequencies(
        filtered_df_interactions,
        item_summary=True,
        user_summary=True,
        show_plots=False,
    )
