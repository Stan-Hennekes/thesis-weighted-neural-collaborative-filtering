"""
Training of the NeuMF (and NeuWMF) model. Based on the following script:
https://github.com/hexiangnan/neural_collaborative_filtering.
"""

import argparse
from time import time

from keras.layers import (
    Dense,
    Embedding,
    Input,
    Dropout,
    Flatten,
    concatenate,
    multiply,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from eals_model.eals_functions import item_pop_weights
from neumf_model.evaluate_neumf import *
from neumf_model.utils_neumf import *

LOGGER = logging.getLogger(__name__)

comment = ""


def parse_args():
    """
    Contains arguments taken by the NeuMF model that do not change often.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modeldir", type=str, default="models", help="models directory"
    )

    # General parameters.
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate.")
    parser.add_argument(
        "--learner",
        type=str,
        default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd",
    )

    # GMF set up.
    parser.add_argument(
        "--n_emb", type=int, default=20, help="embedding size for the GMF part."
    )
    parser.add_argument(
        "--reg_emb_mf",
        type=float,
        default=0.02,
        help="l2 regularization for the GMF part.",
    )

    # MLP set up.
    parser.add_argument(
        "--layers",
        type=str,
        default="[32,32,16,8]",
        help="layer architecture. The first elements is used for the embedding \
        layers for the MLP part and equals n_emb*2",
    )
    parser.add_argument(
        "--reg_emb_mlp",
        type=float,
        default=0.0005,
        help="l2 regularization for the embedding of MLP part.",
    )
    parser.add_argument(
        "--reg_mlp",
        type=float,
        default=0.0005,
        help="l2 regularization for the MLP part.",
    )
    parser.add_argument(
        "--dropouts",
        type=str,
        default="[0.00001,0.00001,0.00001]",
        help="dropout per dense layer. len(dropouts) = len(layers)-1",
    )

    # Output layer.
    parser.add_argument(
        "--reg_out",
        type=float,
        default=0,
        help="l2 regularization for the output layer.",
    )

    # Pretrained model names.
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="freeze all but the last output layer where \
        weights are combined",
    )
    parser.add_argument(
        "--mf_pretrain",
        type=str,
        default="",
        help="Specify the pretrain model filename for GMF part. \
        If empty, no pretrain will be used",
    )
    parser.add_argument(
        "--mlp_pretrain",
        type=str,
        default="",
        help="Specify the pretrain model filename for MLP part. \
        If empty, no pretrain will be used",
    )

    # Experiment set up.
    parser.add_argument(
        "--validate_every", type=int, default=1, help="validate every n epochs"
    )
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument(
        "--n_neg",
        type=int,
        default=3,
        help="number of negative instances to consider per positive instance.",
    )
    parser.add_argument(
        "--topK",
        type=int,
        default=20,
        help="number of items to retrieve for recommendation.",
    )

    return parser.parse_args()


def NeuMF(
    n_users,
    n_items,
    n_emb,
    layer_sizes,
    dropouts,
    reg_emb_mf,
    reg_emb_mlp,
    reg_mlp,
    reg_out,
):
    """
    Define NeuMF model in keras.

    Consists of 2 parts: GMF (simple multiplication of item and user embedding) and MLP (several relu layers on top of
    the embeddings). Both are combined in the final layer of the NeuMF model to a score in [0,1] for each user-item
    combination.

    Args:
        n_users (int): Number of users in train.
        n_items (int): Number of items in train.
        n_emb (int): Number of dimensions in GMF part.
        layer_sizes (list[int]): List of number of elements in each layer of MLP.
        dropouts (list[float]): List of amount of dropout in each layer.
        reg_emb_mf (float): Amount of regularization in MF embedding.
        reg_emb_mlp (float): Amount of regularization in MLP embedding.
        reg_mlp (float): Amount of regularization in MLP embedding.
        reg_out (float): Amount of regularization in output layer.

    Returns:
        Keras model: NeuMF model.

    """

    # Number of layers in the MLP.
    num_layer = len(layer_sizes)

    user = Input(shape=(1,), dtype=np.int32, name="user_input")
    item = Input(shape=(1,), dtype=np.int32, name="item_input")

    # User and item embeddings.
    MF_Embedding_User = Embedding(
        input_dim=n_users,
        output_dim=n_emb,
        name="mf_user_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mf),
        input_length=1,
    )
    MF_Embedding_Item = Embedding(
        input_dim=n_items,
        output_dim=n_emb,
        name="mf_item_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mf),
        input_length=1,
    )

    MLP_Embedding_User = Embedding(
        input_dim=n_users,
        output_dim=int(layer_sizes[0] / 2),
        name="mlp_user_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mlp),
        input_length=1,
    )
    MLP_Embedding_Item = Embedding(
        input_dim=n_items,
        output_dim=int(layer_sizes[0] / 2),
        name="mlp_item_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mlp),
        input_length=1,
    )

    # GMF part.
    mf_user_latent = Flatten()(MF_Embedding_User(user))
    mf_item_latent = Flatten()(MF_Embedding_Item(item))
    mf_vector = multiply([mf_user_latent, mf_item_latent])

    # MLP part.
    mlp_user_latent = Flatten()(MLP_Embedding_User(user))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item))
    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(
            layer_sizes[idx],
            activation="relu",
            kernel_regularizer=l2(reg_mlp),
            name="layer{}".format(idx),
        )
        mlp_vector = layer(mlp_vector)
        mlp_vector = Dropout(dropouts[idx - 1])(mlp_vector)

    predict_vector = concatenate([mf_vector, mlp_vector])

    # Final prediction layer.
    prediction = Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=l2(reg_out),
        kernel_initializer="lecun_uniform",
        name="prediction",
    )(predict_vector)

    # Model.
    model = Model(inputs=[user, item], outputs=prediction)

    return model


def NeuMF_use_info(
    n_users,
    n_items,
    n_emb,
    layer_sizes,
    dropouts,
    reg_emb_mf,
    reg_emb_mlp,
    reg_mlp,
    reg_out,
):
    """
    Define NeuMF model in keras with extra features on users (household type) and items (bio, brand tier).

    Consists of 2 parts: GMF (simple multiplication of item and user embedding) and MLP (several relu layers on top of
    the embeddings). Both are combined in the final layer of the NeuMF model to a score in [0,1] for each user-item
    combination.

    Args:
        n_users (int): Number of users in train.
        n_items (int): Number of items in train.
        n_emb (int): Number of dimensions in GMF part.
        layer_sizes (list[int]): List of number of elements in each layer of MLP.
        dropouts (list[float]): List of amount of dropout in each layer.
        reg_emb_mf (float): Amount of regularization in MF embedding.
        reg_emb_mlp (float): Amount of regularization in MLP embedding.
        reg_mlp (float): Amount of regularization in MLP embedding.
        reg_out (float): Amount of regularization in output layer.

    Returns:
        Keras model: NeuMF model.

    """

    # Number of layers in the MLP.
    num_layer = len(layer_sizes)

    user = Input(shape=(1,), dtype=np.int32, name="user_input")
    item = Input(shape=(1,), dtype=np.int32, name="item_input")

    tier = Input(shape=(1,), dtype=np.int32, name="tier_input")
    bio = Input(shape=(1,), dtype=np.float32, name="bio_input")
    household_type = Input(shape=(1,), dtype=np.float32, name="household_type_input")

    # User and item embeddings.
    MF_Embedding_User = Embedding(
        input_dim=n_users,
        output_dim=n_emb,
        name="mf_user_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mf),
        input_length=1,
    )
    MF_Embedding_Item = Embedding(
        input_dim=n_items,
        output_dim=n_emb,
        name="mf_item_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mf),
        input_length=1,
    )

    MLP_Embedding_User = Embedding(
        input_dim=n_users,
        output_dim=int(layer_sizes[0] / 2),
        name="mlp_user_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mlp),
        input_length=1,
    )

    MLP_Embedding_User_Info = Embedding(
        input_dim=5,
        output_dim=5,
        name="mlp_user_info_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mlp),
        input_length=1,
    )

    MLP_Embedding_Item = Embedding(
        input_dim=n_items,
        output_dim=int(layer_sizes[0] / 2),
        name="mlp_item_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mlp),
        input_length=1,
    )

    MLP_Embedding_Item_Info = Embedding(
        input_dim=3,
        output_dim=3,
        name="mlp_item_info_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(reg_emb_mlp),
        input_length=1,
    )

    # GMF part.
    mf_user_latent = Flatten()(MF_Embedding_User(user))
    mf_item_latent = Flatten()(MF_Embedding_Item(item))
    mf_vector = multiply([mf_user_latent, mf_item_latent])

    # MLP part.
    mlp_user_latent = Flatten()(MLP_Embedding_User(user))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item))
    # Add extra info as well
    mlp_item_info = Flatten()(MLP_Embedding_Item_Info(tier))
    mlp_user_info = Flatten()(MLP_Embedding_User_Info(household_type))
    mlp_vector = concatenate(
        [mlp_user_latent, mlp_item_latent, mlp_item_info, mlp_user_info, bio]
    )

    for idx in range(1, num_layer):
        layer = Dense(
            layer_sizes[idx],
            activation="relu",
            kernel_regularizer=l2(reg_mlp),
            name="layer{}".format(idx),
        )
        mlp_vector = layer(mlp_vector)
        mlp_vector = Dropout(dropouts[idx - 1])(mlp_vector)

    predict_vector = concatenate([mf_vector, mlp_vector])

    # Final prediction layer.
    prediction = Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=l2(reg_out),
        kernel_initializer="lecun_uniform",
        name="prediction",
    )(predict_vector)

    # Model.
    model = Model(inputs=[user, item, tier, bio, household_type], outputs=prediction)

    return model


def NeuMF_model(
    train,
    test,
    loss,
    user_lookup,
    item_lookup,
    use_pos_weights=True,
    use_neg_weights=True,
    c_0=1,
    eta=0.5,
    n_negatives=3,
    lambda_emb_GMF=0,
    lambda_emb_MLP=0,
    lambda_MLP=0,
    lambda_out=0,
    use_info=False,
):
    """
    Train NeuMF model.

    Args:
        train (csr_matrix): Sparse item-user matrix containing train purchases.
        test (csr_matrix): Sparse item-user matrix containing test purchases.
        loss (str): which loss function to use: 'l2_loss' or 'binary_loss'.
        item_lookup (DataFrame): Table containing item IDs and extra info to use as features in model.
        user_lookup (DataFrame): Table containing user IDs and extra info to use as features in model.
        use_pos_weights (bool): Whether or not the values in train matrix are taken as labels,
            or just a 1 for every user-item with a purchase.
        use_neg_weights (bool): Whether or not to sample negatives according to the weights in c.
        c_0 (float): Parameter about negative sampling rate.
        eta (float): Parameter about negative sampling rate.
        n_negatives (int): Number of negative instances to sample per positive user-item interaction.
        lambda_emb_GMF (float): Amount of regularization in MF embedding.
        lambda_emb_MLP (float): Amount of regularization in MLP embedding.
        lambda_MLP (float): Amount of regularization in MLP layers.
        lambda_out (float): Amount of regularization in output layer.
        use_info (bool): Whether or not to include extra user and item info in the model.

    Returns:
        Tuple[np.array, np.array, dict, list, list, list]: Predicted relevance scores of the last iteration and the best
        iteration in terms of NDCG (of new items), dictionary with model settings and loss, NDCG and NDCG new values
        for each iteration.

    """

    c_negative_sampling_weights = item_pop_weights(train, c_0, eta)
    train, test = (
        train.astype("float").todok(),
        test.astype("float").todok(),
    )

    args = parse_args()

    modelfname = "neumf_model" + "_" + args.learner + comment + ".h5"
    modelpath = "neumf_model\\" + str(os.path.join(args.modeldir, modelfname))
    resultsdfpath = "neumf_model\\" + os.path.join(args.modeldir, "results_df.p")
    predictionspath = "scratchpad\\Models\\Bhat_NeuMF"

    n_users, n_items = train.shape
    LOGGER.info("(n_users, n_items): %s", train.shape)

    # Build model.
    if use_info:
        model = NeuMF_use_info(
            n_users,
            n_items,
            args.n_emb,
            eval(args.layers),
            eval(args.dropouts),
            lambda_emb_GMF,  # args.reg_emb_mf,
            lambda_emb_MLP,  # args.reg_emb_mlp,
            lambda_MLP,  # args.reg_mlp,
            lambda_out,
        )
    else:
        model = NeuMF(
            n_users,
            n_items,
            args.n_emb,
            eval(args.layers),
            eval(args.dropouts),
            lambda_emb_GMF,  # args.reg_emb_mf,
            lambda_emb_MLP,  # args.reg_emb_mlp,
            lambda_MLP,  # args.reg_mlp,
            lambda_out,
        )

    # Use Adam learner, different losses.
    if loss == "l2_loss":
        model.compile(optimizer=Adam(lr=args.lr), loss=l2_loss())
    elif loss == "binary_loss":
        model.compile(optimizer=Adam(lr=args.lr), loss="binary_crossentropy")
    else:
        print("No valid loss function")

    # Iterate over models.
    best_hr, best_ndcg, best_iter = 0, 0, 0
    losses = np.zeros(args.epochs)
    NDCGs = np.zeros(args.epochs)
    NDCGs_new = np.zeros(args.epochs)

    if use_info:
        item_lookup["art_p_brand_tier_cat"] = (
            item_lookup["art_p_brand_tier"].astype("category").cat.codes
        )
        item_lookup["bio"] = item_lookup["art_p_art_name"].str.contains("bio")
        user_lookup["household_type_cat"] = (
            user_lookup["cust_household_type"].astype("category").cat.codes
        )

    for epoch in range(1, args.epochs + 1):
        t1 = time()
        LOGGER.info("start sampling negative instances")
        if use_info:
            (
                user,
                item,
                labels,
                tier,
                bio,
                household_type,
            ) = get_train_instances_with_info(
                train,
                n_items,
                n_negatives,
                user_lookup,
                item_lookup,
                c=c_negative_sampling_weights,
                use_pos_weights=use_pos_weights,
                use_neg_weights=use_neg_weights,
            )
        else:
            user, item, labels = get_train_instances(
                train,
                n_items,
                n_negatives,
                c=c_negative_sampling_weights,
                use_pos_weights=use_pos_weights,
                use_neg_weights=use_neg_weights,
            )
        LOGGER.info("Done sampling negative instances")
        # Fit model.
        if use_info:
            hist = model.fit(
                [
                    user.astype("int"),
                    item.astype("int"),
                    tier.astype("int"),
                    bio.astype("float"),
                    household_type.astype("int"),
                ],
                labels.astype("float"),
                # steps_per_epoch = 1,
                batch_size=args.batch_size,
                epochs=1,
                verbose=0,
                shuffle=True,
            )
        else:
            hist = model.fit(
                [user.astype("int"), item.astype("int")],
                labels.astype("float"),
                # steps_per_epoch = 1,
                batch_size=args.batch_size,
                epochs=1,
                verbose=0,
                shuffle=True,
            )
        t2 = time()
        args.validate_every = args.epochs
        if epoch % args.validate_every == 0:
            NDCG_new, HR_new, NDCG, HR, B_hat = evaluate_model(
                model, test, train, args.topK, item_lookup, user_lookup, use_info
            )
            loss = hist.history["loss"][0]
            LOGGER.info(
                "Iteration {}: {:.2f}s, HR new = {:.4f}, NDCG new = {:.4f}, loss = {:.4f}, "
                "HR = {:.4f}, NDCG = {:.4f}, validated in {:.2f}s".format(
                    epoch, t2 - t1, HR_new, NDCG_new, loss, HR, NDCG, time() - t2
                )
            )
            losses[epoch - 1] = loss
            NDCGs[epoch - 1] = NDCG
            NDCGs_new[epoch - 1] = NDCG_new
            if NDCG_new > best_ndcg or (best_ndcg == 0 and epoch == args.epochs):
                best_hr, best_ndcg, best_iter, train_time = (
                    HR_new,
                    NDCG_new,
                    epoch,
                    t2 - t1,
                )
                best_B_hat = B_hat
                if bool(args.save_model):
                    model.save_weights(modelpath, overwrite=True)
                    pd.DataFrame(B_hat).to_pickle(predictionspath)

    LOGGER.info(
        "End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}. ".format(
            best_iter, best_hr, best_ndcg
        )
    )
    if bool(args.save_model):
        LOGGER.info("The best NeuMF model is saved to {}".format(modelpath))
        save_neumf_model_results(
            modelfname,
            NDCG,
            NDCG_new,
            best_iter,
            train_time,
            use_pos_weights,
            use_neg_weights,
            loss,
            use_info,
            resultsdfpath,
        )

    return B_hat, best_B_hat, args, losses, NDCGs, NDCGs_new
