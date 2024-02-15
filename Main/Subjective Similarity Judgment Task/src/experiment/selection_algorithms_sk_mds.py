"""
    
    This file contains code corresponding to a generalized version of Algorithm 1 from Section 3.4    
    The parametrization of the crowd oracle, metric learner, and body selector is intended for the ease of 
    Iterating over the different selection strategies tested in this paper.

"""
import time
import typing
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

import random
import numpy as np
from numpy import linalg as la
from sklearn.manifold import MDS

from experiment import body_selector_manager
from experiment.sim_tracker import SimTracker
from logger.log_writer import LogWriter
from selector_type import SelectorType
from util import matrix_utils



def generate_initial_constraints_from_oracle_result(oracle_sorted_tuples: List[tuple]) -> List[tuple]:
    initial_constraints = []
    for oracle_sorted_tuple in oracle_sorted_tuples:
        for i in range(len(oracle_sorted_tuple) - 2):
            pairwise_comparison = oracle_sorted_tuple[0], oracle_sorted_tuple[i + 1], oracle_sorted_tuple[i + 2]
            initial_constraints.append(pairwise_comparison)
    return initial_constraints


def generate_previous_selections_from_initial_constraints(initial_constraints: list) -> Dict[int, list]:
    previous_selections = defaultdict(list)

    for constraint in initial_constraints:
        head, body = constraint[0], constraint[1:]
        previous_selections[head].append(constraint)

    return previous_selections


def invert_matrix_similarity(sim_matrix: np.array) -> np.array:
    """Take a (dis)similarity matrix and make it into the inverse. Assume values between 0 and 1.
    """
    inverted_matrix = [[None for _ in range(len(sim_matrix[0]))] for _ in range(len(sim_matrix))]
    for x in range(len(sim_matrix)):
        for y in range(len(sim_matrix[0])):
            if sim_matrix[x][y] is not None:
                inverted_matrix[x][y] = 1 - sim_matrix[x][y]
                inverted_matrix[y][x] = 1 - sim_matrix[x][y]
    return np.array(inverted_matrix)


class SelectionAlgorithm:
    """Contains related objects and state for the selection algorithm.
    """

    def __init__(self, init_embedding: np.array, burn_in_iterations: int, run_iterations: int,
                 oracle: typing.Callable, log_writer: LogWriter, tuple_size=5, verbose=False):
        """
        Args:
            init_embedding (numpy.array): Initial Nxd embedding from which an initial similarity matrix can be calculated.
            burn_in_iterations (int): Initial number of burn-in iterations to initialize the similarity matrix.
            run_iterations (int): Number of iterations to run the selection algorithm for.
            oracle (typing.Callable): An oracle function that orders the elements of the passed tuples.
            verbose (bool): Verbose output?
        """
        self.init_sim_matrix = init_embedding
        self.candidates = list(range(len(init_embedding)))  # candidates as a list of numbers
        self.burn_in_iterations = burn_in_iterations
        self.run_iterations = run_iterations
        self.oracle = oracle
        self.log_writer = log_writer
        self.tuple_size = tuple_size
        self.verbose = verbose

    def generate_sim_matrix(self, sim_matrix_inter) -> np.array:
        # calculate the sim matrix
        sim_matrix = [[None for _ in range(len(self.init_sim_matrix[0]))] for _ in range(len(self.init_sim_matrix))]
        for x in range(len(sim_matrix)):
            for y in range(len(sim_matrix[0])):
                if x == y:
                    sim_matrix[x][y] = 0.99999999
                    sim_matrix[y][x] = 0.99999999
                    continue
                # if we have no data at all, set it to 1 (which becomes 0 in dissimilarity)
                if sim_matrix_inter[x][y][0] == 0:
                    # set missing values to 1 so when we get dissimilarity they become 0 and skipped by MDS
                    # this is definitely janky, maybe there's a cleaner way to do it
                    sim_matrix[x][y] = 1
                    sim_matrix[y][x] = 1
                    continue
                # [7,0] This means x,y was compared 7 times but was never close. [15, 0] != [1, 0].
                if sim_matrix_inter[x][y][0] > 0 and sim_matrix_inter[x][y][1] == 0:
                    sim_matrix[x][y] = sim_matrix_inter[x][y][1] + 0.5 / (sim_matrix_inter[x][y][0] + 2)
                    sim_matrix[y][x] = sim_matrix_inter[x][y][1] + 0.5 / (sim_matrix_inter[x][y][0] + 2)
                    continue
                sim_matrix[x][y] = sim_matrix_inter[x][y][1] / (sim_matrix_inter[x][y][0] + 2)
                sim_matrix[y][x] = sim_matrix_inter[x][y][1] / (sim_matrix_inter[x][y][0] + 2)
        """
        print(f"Sim matrix:\n{sim_matrix}")
        """
        return np.array(sim_matrix)

    def generate_intermediate_sim_matrix(self, sim_tracker: SimTracker) -> list:
        """Tracks, for each x, y, the number of times x & y are compared and how many times x & y are closer.

        We use the equation of (number of times when other face was closer)/(number of times faces were compared)
        as described in the return value description.

        Returns:
            list: A 2d array where each (x, y) coordinate contains a 2-element list where the first value [0] is the
            number of times the faces were compared. The second value [1] is the number of times when, when the faces
            were compared, the other face was chosen as closest.
        """
        sim_matrix_inter = [[[0, 0] for _ in range(len(self.init_sim_matrix[0]))]
                            for _ in range(len(self.init_sim_matrix))]
        # target is the stim face - .items() gives key, value as two objects
        for target, responses in sim_tracker.sim_tracker.items():
            for chosen_faces in responses:
                first = chosen_faces[0]
                second = chosen_faces[1]
                # make the denominator
                # count # of times x and y are compared
                sim_matrix_inter[target][first][0] += 1
                sim_matrix_inter[target][second][0] += 1
                # make the matrix symmetrical
                sim_matrix_inter[first][target][0] += 1
                sim_matrix_inter[second][target][0] += 1

                # make the numerator
                # count # of times x & y are closer together
                sim_matrix_inter[target][first][1] += 1
                # make the matrix symmetrical
                sim_matrix_inter[first][target][1] += 1
        """
        print(f"Intermediate sim_matrix:\n")
        for row in sim_matrix_inter:
            for col in row:
                print(col, end=",")
            print()
        """
        return sim_matrix_inter

    def burn_in(self, click_data_list: list, seed=None):
        """Runs the initial burn-in iterations for the selection algorithm.

        Returns:
            oracle_sorted_tuples (list): List of tuples sorted by the oracle.
        """
        if seed:
            rgen = np.random.default_rng(seed)  # enable testing with same burn-ins
        else:
            rgen = np.random.default_rng()
        oracle_sorted_tuples = []
        start_time = time.time()
        for i in range(self.burn_in_iterations):
            target_faces = list(range(len(self.init_sim_matrix)))
            random.shuffle(target_faces)
            for target_face_num in target_faces:
                candidate_tuple = [target_face_num] + list(rgen.choice(self.candidates, self.tuple_size - 1,
                                                                       replace=False))
                while candidate_tuple.count(target_face_num) > 1:  # avoid duplicates in burn-in
                    candidate_tuple = [target_face_num] + list(rgen.choice(self.candidates, self.tuple_size - 1,
                                                                           replace=False))

                response = self.oracle(candidate_tuple, True)
                if response.timed_out:
                    target_faces.append(target_face_num)
                    click_data = response.click_data
                    for click_data_entry in click_data:
                        click_data_entry["iteration"] = -1
                    click_data_list.extend(click_data)
                else:
                    response_tuple = response.response_tuple
                    click_data = response.click_data
                    oracle_sorted_tuples.append(response_tuple)
                    for click_data_entry in click_data:
                        click_data_entry["iteration"] = -1
                    click_data_list.extend(click_data)
        print(f"Burn-in done, time taken: {time.time() - start_time}")
        return oracle_sorted_tuples

    def generate_dissimilarity_matrix(self, sim_tracker: SimTracker) -> np.array:
        intermediate_sim_matrix = self.generate_intermediate_sim_matrix(sim_tracker)
        sim_matrix = self.generate_sim_matrix(intermediate_sim_matrix)
        return invert_matrix_similarity(sim_matrix)

    def run_remainder(self):
        pass

    def selection_algorithm(self, run_data_list: list, run_data: dict, click_data_list: list,
                            body_selector_data_list: list, n_components=5, n_init=100, max_iter=400,
                            target_matrix=None, selector_type=SelectorType.BODY, seed=None):
        """Selection algorithm.
        1:
        For each face pair (x, y), we ask:
            A) How many times are X and Y closer together?
            B) How many times are X and Y compared?
        We then use A/(B+2) as the similarity value. The +2 is there because we implicitly include the times X is
        compared to itself. So, 1 means completely similar and 0 means completely dissimilar.

        2:
        We then flip the sim matrix into a dissimilarity matrix and pass it into scikit-learn's MDS algorithm.

        3:
        After getting the embedding from the MDS algorithm, we run infotuple's body_selector code to choose tuples to
        then show to our oracle (the test subject). The subject orders the faces by similarity and we use this result
        to update our similarity values as in (1).

        Loop 1-3 until max_iterations or target_correlation is hit.

        Returns:
            numpy.array: An embedding that captures the selected ordinal constraints
        """
        if selector_type == SelectorType.RANDOM:
            return self.selection_algorithm_random(run_data_list, run_data, click_data_list, n_components, n_init,
                                                   max_iter, target_matrix)

        if self.verbose:
            Ms, selections, selection_qualities = [], [], []

        # tracks, for each face, which other two faces were chosen as the closest and second closest
        # 0: [(3, 4), (2, 1), (3, 1)]
        sim_tracker = SimTracker()

        max_iterations = 20

        burn_in_sorted_tuples = self.burn_in(click_data_list, seed=seed)
        self.log_writer.write_click_data(click_data_list)

        sim_tracker.add_oracle_result_list(burn_in_sorted_tuples)
        initial_constraints = generate_initial_constraints_from_oracle_result(burn_in_sorted_tuples)

        body_selector = body_selector_manager.BodySelectorManager(self.candidates,
                                                                  self.tuple_size,
                                                                  previous_selections=
                                                                  generate_previous_selections_from_initial_constraints(
                                                                      initial_constraints))
        # use non-metric initially then move to metric after matrix is populated
        initial_mds = MDS(n_components=n_components, dissimilarity="precomputed", normalized_stress=True,
                          metric=False, n_init=n_init, max_iter=max_iter)
        run_mds = MDS(n_components=n_components, dissimilarity="precomputed", normalized_stress=False, metric=True,
                      n_init=n_init, max_iter=max_iter)

        dissim_matrix = self.generate_dissimilarity_matrix(sim_tracker)

        print(f"Dissimilarity matrix for first MDS:\n{dissim_matrix}")
        print(f"Starting burn-in MDS...")
        first_MDS_start_time = time.time()
        M_prime = initial_mds.fit_transform(dissim_matrix)
        print(f"First MDS done, time taken = {time.time() - first_MDS_start_time}")

        for i in range(min(max_iterations, self.run_iterations)):
            iter_start_time = time.time()
            if self.verbose:
                Ms.append(M_prime)
            target_faces = list(range(len(self.init_sim_matrix)))
            random.shuffle(target_faces)
            for target_face_num in target_faces:
                selected_tuple, tuple_qualities = body_selector.run_body_selector_with_previous_selections(
                    M_prime, target_face_num, body_selector_data_list)

                oracle_result = self.oracle(selected_tuple, True)
                print(oracle_result)
                if oracle_result.timed_out:
                    target_faces.append(target_face_num)

                    click_datas = oracle_result.click_data
                    for click_data in click_datas:
                        click_data["iteration"] = i
                    click_data_list.extend(click_datas)
                else:
                    oracle_sorted_tuple = oracle_result.response_tuple
                    sim_tracker.add_oracle_result(oracle_sorted_tuple)

                    click_datas = oracle_result.click_data
                    for click_data in click_datas:
                        click_data["iteration"] = i
                    click_data_list.extend(click_datas)

                    if self.verbose:
                        selections.append(selected_tuple)
                        selection_qualities.append(tuple_qualities)

            prev_dissim_matrix = deepcopy(dissim_matrix)
            dissim_matrix = self.generate_dissimilarity_matrix(sim_tracker)

            body_selector_iter_time = time.time() - iter_start_time
            print(f"Total time for body selector this iteration was {body_selector_iter_time}")
            print(f"Starting MDS...")
            MDS_start_time = time.time()
            M_prime = run_mds.fit_transform(dissim_matrix, M_prime)
            print(f"MDS done, time taken = {time.time() - MDS_start_time}")

            run_data["iteration"] = i
            if target_matrix is not None:
                run_data["correlation_embedding"] = matrix_utils.correlation(
                    matrix_utils.get_output_matrix_from_embedding(M_prime, len(M_prime)),
                    target_matrix)[0][1]
                run_data["correlation_sim_matrix"] = matrix_utils.correlation(dissim_matrix, target_matrix)[0][1]
            run_data["change_sim_matrix"] = la.norm(dissim_matrix.flatten() - prev_dissim_matrix.flatten())
            run_data_list.append(deepcopy(run_data))

            self.log_writer.write_all(click_data_list, run_data_list, M_prime, dissim_matrix, len(self.init_sim_matrix))

            if run_data["change_sim_matrix"] < 0.5:
                break

        """
        1. Find the zeros in the sim matrix (x,y)
        Run two trials - one where x is target, y is candidate, another where y is target and x is candidate
        The rest of the faces come from the body selector.
        Mix the remaining trials. Do all x,y first, then do all y,x.
        """
        # Fill out the zeroes in the matrix
        remaining_pairs = []

        i += 1  # treat the zero filling as one "iteration"
        for x in range(len(dissim_matrix)):
            for y in range(len(dissim_matrix[0])):
                if x == y:
                    continue
                if dissim_matrix[x][y] == 0:
                    # this will also append (y, x) which is what we want
                    remaining_pairs.append((x, y))
        for face_pair in remaining_pairs:
            target_face_num = face_pair[0]
            selected_tuple, tuple_qualities = body_selector.run_body_selector_with_previous_selections(
                M_prime, target_face_num, body_selector_data_list, fixed_candidates=[face_pair[1]])

            oracle_result = self.oracle(selected_tuple, True)

            if oracle_result.timed_out:
                remaining_pairs.append(target_face_num)

                click_datas = oracle_result.click_data
                for click_data in click_datas:
                    click_data["iteration"] = i
                click_data_list.extend(click_datas)
            else:
                oracle_sorted_tuple = oracle_result.response_tuple
                sim_tracker.add_oracle_result(oracle_sorted_tuple)

                click_datas = oracle_result.click_data
                for click_data in click_datas:
                    click_data["iteration"] = i
                click_data_list.extend(click_datas)

                if self.verbose:
                    selections.append(selected_tuple)
                    selection_qualities.append(tuple_qualities)

        # print(f"Total time for body selector this iteration was {body_selector_iter_time}")

        prev_dissim_matrix = deepcopy(dissim_matrix)
        dissim_matrix = self.generate_dissimilarity_matrix(sim_tracker)

        print(f"Starting MDS...")
        MDS_start_time = time.time()
        M_prime = run_mds.fit_transform(dissim_matrix, M_prime)
        print(f"MDS done, time taken = {time.time() - MDS_start_time}")

        run_data["iteration"] = i
        if target_matrix is not None:
            run_data["correlation_embedding"] = matrix_utils.correlation(
                matrix_utils.get_output_matrix_from_embedding(M_prime, len(M_prime)),
                target_matrix)[0][1]
            run_data["correlation_sim_matrix"] = matrix_utils.correlation(dissim_matrix, target_matrix)[0][1]
        run_data["change_sim_matrix"] = la.norm(dissim_matrix.flatten() - prev_dissim_matrix.flatten())
        run_data_list.append(deepcopy(run_data))

        if self.verbose:
            Ms.append(M_prime)
            return Ms, (initial_constraints, selections), selection_qualities

        return M_prime, dissim_matrix

    def selection_algorithm_random(self, run_data_list: list, run_data: dict, click_data_list: list,
                                   n_components=5, n_init=100, max_iter=400, target_matrix=None, seed=None):
        """Selection algorithm which uses random selection.
        Returns:
            numpy.array: An embedding that captures the selected ordinal constraints
        """
        if self.verbose:
            Ms, selections, selection_qualities = [], [], []

        # tracks, for each face, which other two faces were chosen as the closest and second closest
        # 0: [(3, 4), (2, 1), (3, 1)]
        sim_tracker = SimTracker()

        max_iterations = 20

        burn_in_sorted_tuples = self.burn_in(click_data_list, seed=seed)
        sim_tracker.add_oracle_result_list(burn_in_sorted_tuples)
        # use non-metric initially then move to metric after matrix is populated
        initial_mds = MDS(n_components=n_components, dissimilarity="precomputed", normalized_stress=True,
                          metric=False, n_init=n_init, max_iter=max_iter)
        run_mds = MDS(n_components=n_components, dissimilarity="precomputed", normalized_stress=False, metric=True,
                      n_init=n_init, max_iter=max_iter)

        dissim_matrix = self.generate_dissimilarity_matrix(sim_tracker)

        print(f"Dissimilarity matrix for first MDS:\n{dissim_matrix}")
        print(f"Starting burn-in MDS...")
        first_MDS_start_time = time.time()
        M_prime = initial_mds.fit_transform(dissim_matrix)
        print(f"First MDS done, time taken = {time.time() - first_MDS_start_time}")

        for i in range(min(max_iterations, self.run_iterations)):
            for target_face_num in range(len(self.init_sim_matrix)):
                selected_tuple = [target_face_num] + list(
                    np.random.choice(self.candidates, self.tuple_size - 1, replace=False))
                while selected_tuple.count(target_face_num) > 1:  # avoid duplicates in burn-in
                    selected_tuple = [target_face_num] + list(
                        np.random.choice(self.candidates, self.tuple_size - 1, replace=False))

                oracle_result = self.oracle(selected_tuple)

                oracle_sorted_tuple = oracle_result.response_tuple
                sim_tracker.add_oracle_result(oracle_sorted_tuple)

                click_datas = oracle_result.click_data
                for click_data in click_datas:
                    click_data["iteration"] = i
                click_data_list.extend(click_datas)

            prev_dissim_matrix = deepcopy(dissim_matrix)
            dissim_matrix = self.generate_dissimilarity_matrix(sim_tracker)

            print(f"Starting MDS...")
            MDS_start_time = time.time()
            M_prime = run_mds.fit_transform(dissim_matrix, M_prime)
            print(f"MDS done, time taken = {time.time() - MDS_start_time}")

            run_data["iteration"] = i
            if target_matrix is not None:
                run_data["correlation_embedding"] = matrix_utils.correlation(
                    matrix_utils.get_output_matrix_from_embedding(M_prime, len(M_prime)),
                    target_matrix)[0][1]
                run_data["correlation_sim_matrix"] = matrix_utils.correlation(dissim_matrix, target_matrix)[0][1]
            run_data["change_sim_matrix"] = la.norm(dissim_matrix.flatten() - prev_dissim_matrix.flatten())
            run_data_list.append(deepcopy(run_data))
            """
            if run_data["change_sim_matrix"] < 0.5:
                break
            """

        return M_prime, dissim_matrix
