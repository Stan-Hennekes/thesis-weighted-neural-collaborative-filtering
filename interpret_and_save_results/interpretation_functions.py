import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def write_to_file(
    user_lookup, user, level, bought_items, recommended_items, recommended_items_new
):
    """
    Write recommendation for a specific user to a local file.

    Args:
        user_lookup (pd.DataFrame): Table containing decryption of user IDs to customer keys.
        user (int): User ID.
        level (str): Level of aggregation.
        bought_items (list): List of items already bought by this user in the train set.
        recommended_items (list): List of recommended items for this user.
        recommended_items_new: (list) Only recommend items that do not appear in bought_items.

    """

    user_key = str(list(user_lookup["customer"][user_lookup["user_id"] == user])[0])
    df_info = pd.DataFrame(
        [[user_key], [level]], index=["User", "level"], columns=["Value"]
    )

    filename = "scratchpad/Results/Recommendations_output_" + str(user) + ".xlsx"
    bought_df = pd.DataFrame(bought_items)
    with pd.ExcelWriter(filename) as writer:
        df_info.to_excel(writer, sheet_name="info")
        bought_df.to_excel(writer, sheet_name="bought_items")
        recommended_items.to_excel(writer, sheet_name="recommended_items")
        recommended_items_new.to_excel(writer, sheet_name="recommended_items_new")
