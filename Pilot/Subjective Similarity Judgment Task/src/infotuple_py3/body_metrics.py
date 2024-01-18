"""

    This file contains methods for calculating the probability of a tuplewise ranking
    and the information provided by a tuple query. Relevant references describing
    in detail these procedures and the requisite assumptions are found in Sections 3.1 and 3.2.

"""
import time
from copy import deepcopy
from multiprocessing.pool import ThreadPool

import numpy as np
from random import shuffle
from scipy.stats import entropy
from itertools import permutations
from collections import defaultdict
from scipy.spatial.distance import pdist
import math
from functools import reduce

default_times = []
test_times = []


def random_body_selector(head, M, tuples, intermediate_params=None):
    """
    Random tuple body selection, provided only as a baseline for comparison 
    """
    index = np.random.choice(len(tuples))
    return tuples[index], [], [], None


def probability_model(bdist, cdist, mu):
    """
    This is a helper method for the tuplewise probability calculation (Section 3.2)
    It computes the probability of a response for an individual 3-tuple

    bdist: Distance between a and b_i
    cdist: Distance between a and b_{i+1}
    mu: Optional parameter to specify response model
    """
    prob = (mu + cdist ** 2) / (2 * mu + bdist ** 2 + cdist ** 2)
    return prob


def tuple_probability_model(distances, mu):
    """
    This is the tuplewise response model described in (Section 3.2)
    
    Inputs:
        distances: The precomputed set of pairwise distances between a and each body object,
        mu: Optional regularization parameter, set to 0.5 to ignore
    returns:
        tuple_probability: The probability of the specified tuple
    """
    tuple_probability = 1
    for i in range(len(distances) - 1):
        bdist = distances[i]
        cdist = distances[i + 1]
        this_prob = probability_model(bdist, cdist, mu)
        tuple_probability = tuple_probability * this_prob

    return tuple_probability


def mutual_information(X, head, body, n_samples, dist_std, mu):
    """
    This method corresponds to the mutual information calculation specified in Section 3.1
    Specifically, the method returns the result of inputting the method parameters into formula (9).
    
    Inputs:
        X: An Nxd embedding of objects
        head: The head of the tuple comparison under consideration
        body: The body of the tuple comparison under consideration
        n_samples: Number of samples to estimate D_s as described in (A3)
        dist_std: Variance parameter as specified in (A3)
        mu: Optional regularization parameter for the probabilistic response model
    returns:
        information: Mutual information as specified in (9) in Section 3.1

    """
    nrank = math.factorial(len(body))

    first_term_sampled_probabilities = np.zeros((n_samples, nrank))
    second_term_sampled_entropies = []

    for d in range(n_samples):
        p_rab = []

        for permutation in permutations(body, len(body)):
            B = permutation
            head_distances = []

            for i in range(len(body)):
                norm = np.linalg.norm(X[head] - X[B[i]])
                dist = abs(np.random.normal(norm, dist_std))
                head_distances.append(dist)

            p_rab.append(tuple_probability_model(head_distances, mu))

        normalization_array = np.full(len(p_rab), np.sum(p_rab))
        p_rab = p_rab / normalization_array
        first_term_sampled_probabilities[d, :] = p_rab

        sample_entropy = 0
        for p in p_rab:
            if p > 0:
                sample_entropy -= (p * np.log(p))
        second_term_sampled_entropies.append(sample_entropy)

    first_term_expected_probabilities = np.sum(first_term_sampled_probabilities, axis=0) / n_samples
    first_term_expected_entropy = -np.sum(first_term_expected_probabilities * np.log(first_term_expected_probabilities))
    second_term_expected_entropy = np.sum(second_term_sampled_entropies) / len(second_term_sampled_entropies)
    information = first_term_expected_entropy - second_term_expected_entropy

    # print(f"Mutual_information time: {time.time() - start_time}")
    return information


class InfogainsTask:
    def __init__(self, infogains, M, tuples, dist_std, mu):
        self.M = M
        self.tuples = tuples
        self.dist_std = dist_std
        self.mu = mu
        self.infogains = infogains

    def infogains_task(self, i):
        a = self.tuples[i][0]
        B = self.tuples[i][1:]

        self.infogains[i] = mutual_information(self.M, a, B, self.M.shape[0], self.dist_std, self.mu)


def infogains_task_function(i, infogains, M, tuples, dist_std, mu):
    a = tuples[i][0]
    B = tuples[i][1:]

    infogains[i] = mutual_information(M, a, B, M.shape[0], dist_std, mu)


def primal_body_selector_test(head, M, tuples, __, body_selector_data_list,
                              intermediate_params={'mu': 0.05, 'tuple_downsample_rate': 0.01, 'dist_cache': {}}):
    """
    Inner loop of Algorithm 1, this method selects the tuple body that maximizes our mutual information metric
    Used at each algorithm iteration to compute an optimal query to request, as described in section 3.1.

    Inputs:
        head: Index of head objects
        M: Nxd coordinates with respect to which to select an informative query
        tuples: list of possible tuples over M
        __: Placeholder for the list of previous selections to standardize the method header
            across tested selection algorithms. Not used in this method.
    Returns:
    infogains, tuple_probabilities, intermediate_params
        selected_tuple: The selected informative tuple
        infogains: A list of information gains from all tuples,
                    provided for visualization purposes.
        tuple_probabilities: A list of probabilities for each tuple,
                    provided for visualization purposes

    """

    body_selector_data = {}

    start_time = time.time()
    print(f"Starting body selector using downsample rate {intermediate_params['tuple_downsample_rate']}...")
    print(f"Tuple length is {len(tuples)}")

    downsample_rate = intermediate_params['tuple_downsample_rate']
    num_samples = int(len(tuples) * downsample_rate)
    while num_samples < 5:
        downsample_rate *= 10
        num_samples = int(len(tuples) * downsample_rate)
    downsample_indices = np.random.choice(list(range(len(tuples))), num_samples, replace=False)
    print(f"Downsampled tuple length is: {len(downsample_indices)}")
    tuples = [tuples[i] for i in downsample_indices]

    tuple_probabilities = np.ones(len(tuples))
    infogains = np.zeros(len(tuples))

    mu = intermediate_params['mu']

    distances = pdist(M)
    dist_std = np.sqrt(np.var(distances, axis=0))

    start_time_mutual_info_loop = time.time()
    downsample_ratios = [7]
    for i in range(len(tuples)):
        a = tuples[i][0]
        B = tuples[i][1:]

        body_selector_data["tuple_length"] = len(tuples[0])

        for downsample_ratio in downsample_ratios:
            start_time_mutual_info = time.time()
            body_selector_data["downsample_ratio"] = downsample_ratio
            infogains[i] = mutual_information(M, a, B, M.shape[0]//downsample_ratio, dist_std, mu)
            body_selector_data["runtime"] = time.time() - start_time_mutual_info
            body_selector_data["infogain"] = infogains[i]
            body_selector_data_list.append(deepcopy(body_selector_data))

    # print(f"Mutual info total loop time: {time.time() - start_time_mutual_info_loop}")

    selected_tuple = tuples[np.argmax(infogains)]

    print(f"Body selector done. Time taken: {time.time() - start_time}")
    return selected_tuple, infogains, tuple_probabilities, intermediate_params


def primal_body_selector(head, M, tuples, __,
                         intermediate_params={'mu': 0.05, 'tuple_downsample_rate': 0.01, 'dist_cache': {}}):
    """
    Inner loop of Algorithm 1, this method selects the tuple body that maximizes our mutual information metric
    Used at each algorithm iteration to compute an optimal query to request, as described in section 3.1.

    Inputs:
        head: Index of head objects
        M: Nxd coordinates with respect to which to select an informative query
        tuples: list of possible tuples over M
        __: Placeholder for the list of previous selections to standardize the method header
            across tested selection algorithms. Not used in this method.
    Returns:
    infogains, tuple_probabilities, intermediate_params
        selected_tuple: The selected informative tuple
        infogains: A list of information gains from all tuples,
                    provided for visualization purposes.
        tuple_probabilities: A list of probabilities for each tuple,
                    provided for visualization purposes

    """

    start_time = time.time()
    print(f"Starting body selector using downsample rate {intermediate_params['tuple_downsample_rate']}...")
    print(f"Tuple length is {len(tuples)}")

    downsample_rate = intermediate_params['tuple_downsample_rate']
    num_samples = int(len(tuples) * downsample_rate)
    while num_samples < 5:
        downsample_rate *= 10
        num_samples = int(len(tuples) * downsample_rate)
    downsample_indices = np.random.choice(list(range(len(tuples))), num_samples, replace=False)
    print(f"Downsampled tuple length is: {len(downsample_indices)}")
    tuples = [tuples[i] for i in downsample_indices]

    tuple_probabilities = np.ones(len(tuples))
    infogains = np.zeros(len(tuples))

    mu = intermediate_params['mu']

    distances = pdist(M)
    dist_std = np.sqrt(np.var(distances, axis=0))

    start_time_mutual_info_loop = time.time()
    for i in range(len(tuples)):
        a = tuples[i][0]
        B = tuples[i][1:]

        infogains[i] = mutual_information(M, a, B, M.shape[0]//7, dist_std, mu)
    # print(f"Mutual info total loop time: {time.time() - start_time_mutual_info_loop}")

    selected_tuple = tuples[np.argmax(infogains)]

    print(f"Body selector done. Time taken: {time.time() - start_time}")
    return selected_tuple, infogains, tuple_probabilities, intermediate_params


def kalai_probability_model(head, candidate_tuple, M, previous_selections,
                            intermediate_params={'mu': 0.5, 'dist_cache': {}}):
    """
    Tuple probability model for CKL, see (Tamuz et al, 2011)
    """

    prior = 1. / len(M)

    mu = intermediate_params['mu']
    dist_cache = intermediate_params['dist_cache']

    head, b_candidate, c_candidate = candidate_tuple

    previous_probabilities = []

    posterior = np.zeros(len(M))
    b_posterior = np.zeros(len(M))
    c_posterior = np.zeros(len(M))

    for x in range(len(posterior)):
        if x in list(previous_selections.keys()):
            previous_x_selections = previous_selections[x]
            for triple in previous_x_selections:

                a, b, c = triple

                if (a, b) not in list(dist_cache.keys()):
                    dist_cache[(a, b)] = np.sqrt(sum((M[a] - M[b]) ** 2))

                if (a, c) not in list(dist_cache.keys()):
                    dist_cache[(a, c)] = np.sqrt(sum((M[a] - M[c]) ** 2))

                dab = dist_cache[(a, b)]
                dac = dist_cache[(a, c)]

                triple_probability = (dac + mu) / (2 * mu + dab + dac)
                previous_probabilities.append(triple_probability)

            if len(previous_probabilities) == 1:
                posterior[x] = prior * previous_probabilities[0]
            else:
                prod_abc = reduce(lambda x, y: x * y, previous_probabilities)
                posterior[x] = prior * prod_abc

    posterior /= sum(posterior)

    p = 0.

    for x in range(len(M)):

        if (x, b_candidate) not in list(dist_cache.keys()):
            dist_cache[(x, b_candidate)] = np.sqrt(sum((M[x] - M[b_candidate]) ** 2))

        if (x, c_candidate) not in list(dist_cache.keys()):
            dist_cache[(x, c_candidate)] = np.sqrt(sum((M[x] - M[c_candidate]) ** 2))

        dab = dist_cache[(x, b_candidate)]
        dac = dist_cache[(x, c_candidate)]

        p += ((mu + dac) / (2 * mu + dab + dac)) * posterior[x]

        b_posterior[x] = posterior[x] * (dac / (dab + dac))  # returned for entropy calculation
        c_posterior[x] = posterior[x] * (dab / (dab + dac))  # returned for entropy calculation

    return p, posterior, b_posterior, c_posterior, {'mu': mu, 'dist_cache': dist_cache}


def kalai_default_body_selector(head, M, triples, previous_selections,
                                intermediate_params={'mu': 0.5, 'triplet_downsample_rate': 0.1}):
    """
    Selection procedure for CKL, see (Tamuz et al, 2011)
    """

    downsample_indices = np.random.choice(list(range(len(triples))),
                                          int(len(triples) * intermediate_params['triplet_downsample_rate']),
                                          replace=False)
    triples = [triples[i] for i in downsample_indices]

    dist_cache = {}
    mu = intermediate_params['mu']
    cached_params = {'mu': mu, 'dist_cache': dist_cache}

    infogains = np.zeros(len(triples))
    tuple_probabilities = {}
    infogains = []

    for candidate_tuple in triples:
        p, posterior, b_posterior, c_posterior, cached_params = kalai_tuple_probability_model(head, candidate_tuple, M,
                                                                                              previous_selections,
                                                                                              cached_params)

        infogain = entropy(posterior) - p * entropy(b_posterior) - (1 - p) * entropy(c_posterior)
        tuple_probabilities[candidate_tuple] = p

        infogains.append(infogain)

    selected_tuple = tuples[np.argmax(infogains)]

    return selected_tuple, infogains, tuple_probabilities, cached_params
