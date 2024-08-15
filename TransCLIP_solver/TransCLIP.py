from .TransCLIP_utils import *
import time


def TransCLIP_solver(support_features, support_labels, val_features, val_labels, query_features, query_labels,
                     clip_prototypes, initial_prototypes=None, initial_predictions=None, verbose=True):
    """
    This function implement our TransCLIP method and computes the Top-1 test accuracy for a given set of embeddings
    and labels. The computation involves several inputs, including embeddings for shots and validation samples,
    labels for shots, validation, and test samples, and text embeddings represented as clip weights. These inputs are
    provided as PyTorch tensors.
    
    |S|: total number of shots. |Q|: number of test samples. d: dimensionality size of the feature space. K: number of classes.
    
    Parameters:
    - support_features (torch.Tensor): A tensor of shape (|S|, d) containing embeddings for each shot.
    - support_labels (torch.Tensor): A tensor of shape (|S|, K) containing labels for each shot.
    - val_features (torch.Tensor): A tensor of shape (min(4*K, |S|), d) containing embeddings for validation samples.
    - val_labels (torch.Tensor): A tensor of shape (min(4*K, |S|)) containing labels for validation samples.
    - query_features (torch.Tensor): A tensor of shape (|Q|, d) containing embeddings for query samples.
      '|Q|' represents the number of query samples.
    - query_labels (torch.Tensor): A tensor of shape (|Q|) containing labels for query samples, used solely for evaluation.
    - clip_prototypes (torch.Tensor): A tensor of shape (d, K) containing text embeddings for each class.
    - initial_prototypes (torch.Tensor): A tensor of shape (d, K) containing precomputed class prototypes.
      If initial_prototypes is None, the clip_prototypes are used instead.
    - initial_predictions (torch.Tensor) : A tensor of shape (|Q|, K) that contains query predictions (already softmaxed)
      from another method. Can be used when a method does not compute any class prototypes. If None, clip_prototypes are
      used instead to compute initial pseudo-labels.
    - verbose (bool): indicate if the accuracies should be printed.

    Returns:
    - assignments z: A Tensor of shape (|Q|, K) containing the query set predictions.
    - float: The Top-1 accuracy on the test set, evaluated using the specified embeddings and labels.
    """

    start_time = time.time()

    ################
    # General init #
    ################

    K = len(torch.unique(query_labels))
    d = query_features.size(1)
    num_samples = query_features.size(0)

    y_hat, query_features, query_labels, val_features, val_labels, support_features, support_labels, neighbor_index,acc_base_zs = \
        prepare_objects(query_features, query_labels,
                     val_features, val_labels,
                     support_features, support_labels,
                     clip_prototypes, initial_prototypes,
                     initial_predictions, verbose=verbose)

    max_iter = 10  # number of iterations
    std_init = 1 / d
    n_neighbors = 3
    best_val = -1  # to keep track in few-shot
    test_acc_at_best_val = -1  # to keep track in few-shot

    # Get hyperparameter ranges and prepare shots if any

    lambda_value, gamma_list, support_features, support_labels = get_parameters(support_labels, support_features)

    ##########
    # Z init #
    ##########

    y_hat, z = init_z(y_hat, softmax=False if initial_predictions is not None else True)

    ###########
    # MU init #
    ###########

    mu = init_mu(K, d, z, query_features, support_features, support_labels)

    ##############
    # SIGMA init #
    ##############

    std = init_sigma(d, std_init)

    adapter = Gaussian(mu=mu, std=std).cuda()

    ###################
    # Affinity matrix #
    ###################

    W = build_affinity_matrix(query_features, support_features, num_samples, n_neighbors)

    # Iterate over gamma (gamma = [0] in zero-shot):

    for idx, gamma_value in enumerate(gamma_list):

        for k in range(max_iter + 1):
            gmm_likelihood = adapter(query_features, no_exp=True)

            ############
            # Z update #
            ############

            new_z = update_z(gmm_likelihood, y_hat, z, W, lambda_value, n_neighbors, support_labels)[0:num_samples]
            z = new_z
            if k == max_iter:  # STOP
                acc = cls_acc(z, query_labels)
                if support_features is not None:  # Few-shot : validate gamma
                    acc_val = cls_acc(z[neighbor_index, :], val_labels)
                    if acc_val > best_val:
                        best_val = acc_val
                        test_acc_at_best_val = acc

                else:
                    acc = cls_acc(z, query_labels)
                    if verbose:
                        print("\n**** VLM+TransCLIP's test accuracy: {:.2f} ****\n".format(acc))
                break

            #############
            # MU update #
            #############

            adapter = update_mu(adapter, gamma_value, query_features, z, support_features, support_labels)

            ################
            # SIGMA update #
            ################

            adapter = update_sigma(adapter, gamma_value, query_features, z, support_features, support_labels)

        if support_features is not None:
            if verbose:
                print("{}/{} TransCLIP's test accuracy: {:.2f} on test set @ best validation accuracy ({:.2f})".format(
                    idx+1, len(gamma_list), test_acc_at_best_val, best_val))
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return z, test_acc_at_best_val, acc_base_zs, acc
