import numpy as np
from sklearn.metrics import accuracy_score
### TODO: import any other packages you need for your solution
import cvxpy as cp


# --- Task 1 ---
class MyDecentralized:
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        # self.W = None   # shape (K, M)
        # self.b = None   # shape (K,)



    def train(self, trainX, trainY):
        ''' Task 1
            TODO: train a multi-class linear classifier using LP/ILP.
                  Store learned parameters you will use in predict().
        '''
        N, M = trainX.shape

        pass

    def predict(self, testX):
        ''' Task 1
            TODO: predict class labels for the input data (testX) using the trained classifier
        '''
        # predY = ...
        return predY

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)
        return accuracy


##########################################################################
# --- Task 2 & Task 3 ---
##########################################################################
# --- Task 2 & Task 3 ---
class MyFeatureCompression:
    def __init__(self, K):
        """
        Args:
            K (int): number of classes.
        Notes:
            You may add any state you need (e.g., a base classifier, search grids, bit candidates).
            The project does not constrain the quantizer design; document your choices and bit accounting.
        """
        self.K = K  # number of classes
        # TODO: add any state you need (e.g., bit candidates, a base classifier)

    def run_centralized(self, trainX, trainY, valX, valY, testX, testY, B_tot_list):
        """
        Task 2 (Centralized compression)

        What this function should do (high level, quantizer-agnostic):
        - Assume a single centralized encoder sees all M features.
        - For each total budget in B_tot_list (measured in bits per image),
          produce a quantized representation according to your chosen quantizer,
          train a model (using train data), optionally use validation only for
          model/quantizer hyperparameter selection, and evaluate test accuracy
          on quantized test inputs.

        Args:
            trainX, trainY: training data/labels.
            valX,   valY  : validation data/labels (for model/quantizer selection only; do not touch test labels).
            testX,  testY : test data/labels (final reporting only).
            B_tot_list (Iterable[int]): list of total budgets (bits per image) to evaluate
                in the centralized setting. Example: [784, 1568, 2352, ...] corresponds to
                roughly 1/2/3/... bits per feature if you choose a uniform scalar design
                with M=784. You may implement any centralized quantizer; just ensure your
                bit accounting is clear.

        Returns:
            dict with keys:
                'B_tot'         : list[int], the budgets evaluated (bits per image)
                'test_accuracy' : list[float], test accuracy at each budget

        Notes:
            - The specification does NOT require a particular quantizer. If you use a design
              that needs data-derived parameters (e.g., ranges, codebooks), estimate them from
              training data only (no test leakage).
            - Plotting: this output is used for "accuracy vs B_tot" and to compare against Task 1.
        """
        result = {'B_tot': [], 'test_accuracy': []}
        return result

    def run_decentralized_per_sensor(self, train_blocks, val_blocks, test_blocks, trainY, valY, testY, k_list):
        """
        Task 3.1 (Decentralized, fixed per-sensor budget)

        What this function should do:
        - There are 4 sensors; sensor s observes its feature block (N x d_s).
        - For each per-sensor budget k in k_list (bits per image per sensor),
          design/apply a per-sensor quantizer, concatenate the quantized blocks,
          train on quantized train, optionally use validation only for selection,
          and report test accuracy.

        Args:
            train_blocks, val_blocks, test_blocks: lists of 4 arrays, each [N x d_s],
                corresponding to the four non-overlapping quadrants (sensors).
            trainY, valY, testY: labels for train/val/test.
            k_list (Iterable[int]): list of per-sensor budgets (bits per image per sensor).
                Interpretation is up to your quantizer. A common choice is to derive a
                per-feature bit-depth b_s from k and d_s (e.g., b_s ≈ floor(k / d_s)), but
                this is not mandated; any linear-programming-consistent approach is acceptable
                as long as you document the bit accounting.

        Returns:
            dict with keys:
                'k'             : list[int], the per-sensor budgets evaluated
                'test_accuracy' : list[float], test accuracy at each k
                'b_s'           : list[tuple], optional record of per-sensor bit-depths or
                                   other allocation details per point (for reporting)

        Notes:
            - Keep train/val/test strict: use validation for selection only; do not use test
              information during training or allocation decisions.
            - Plotting: used for "accuracy vs k" and to compare with centralized at matched B_tot.
        """
        result = {'k': [], 'test_accuracy': [], 'b_s': []}
        return result

    def run_decentralized_total(self, train_blocks, val_blocks, test_blocks, trainY, valY, testY, B_tot_list):
        """
        Task 3.2 (Decentralized, fixed total budget)

        What this function should do:
        - Each budget B in B_tot_list is a total bit budget (bits per image) shared across 4 sensors.
        - For each B, explore one or more decentralized bit-allocation strategies across sensors
          (e.g., (b1,b2,b3,b4) if using scalar bit-depths; or any other quantizer-specific allocation),
          use validation accuracy to choose the best allocation, and report test accuracy for that choice.

        Args:
            train_blocks, val_blocks, test_blocks: lists of 4 arrays, each [N x d_s].
            trainY, valY, testY: labels for train/val/test.
            B_tot_list (Iterable[int]): list of total budgets (bits per image) to evaluate
                in the decentralized setting. For example, with scalar bit-depths one might
                constrain sum_s d_s * b_s <= B; but you may implement any decentralized quantizer,
                provided the total-bit accounting is clear and comparable.

        Returns:
            dict with keys:
                'B_tot'          : list[int], the budgets evaluated (bits per image)
                'test_accuracy'  : list[float], test accuracy at each budget
                'best_allocation': list[tuple], a record of the chosen allocation per B
                                    (e.g., (b1,b2,b3,b4) or any quantizer-specific summary)

        Notes:
            - Use validation ONLY to choose among candidate allocations or hyperparameters.
              Test is for final reporting.
            - Plotting: used for "accuracy vs B_tot" and for the centralized vs decentralized overlay.
        """
        result = {'B_tot': [], 'test_accuracy': [], 'best_allocation': []}
        return result



##########################################################################
##########################################################################
# --- Task 3.3 ---
class MyTargetAllocator:
    def __init__(self, K):
        self.K = K  # number of classes
        # TODO: add any state you need

    def minimal_bits_centralized(self, feature_compressor, trainX, trainY, valX, valY, testX, testY, alpha, B_grid):
        """
        Task 3.3 (Centralized)

        Goal:
            Given a target test accuracy α (e.g., 0.7, 0.8, 0.9), find the minimal total bit budget
            B (bits/image) so that your centralized formulation achieves test accuracy ≥ α.

        Allowed approaches (your choice, consistent with the guidelines):
            • Outer-search approach: use an outer search over candidate budgets and, for each,
              solve/evaluate your centralized formulation; pick the smallest B achieving ≥ α.
              In this case, `B_grid` provides the candidate budgets you intend to try (e.g., [784, 1568, 2352, 3136]).
            • Direct optimization approach: encode the minimal-bits objective directly in an LP/ILP/MILP
              that enforces accuracy ≥ α (as you define it) and solve for B. In this case, `B_grid` may be
              ignored or used as a search scaffold/initialization if helpful.

        Args:
            feature_compressor: an object exposing your centralized pipeline (e.g., MyFeatureCompression) if you
                choose to implement the outer-search approach. For a direct optimization approach, you may ignore it.
            trainX, trainY, valX, valY, testX, testY:
                datasets (keep train/val/test strict; no test leakage in model/quantizer selection).
            alpha (float): target test accuracy in [0,1].
            B_grid (Iterable[int]): candidate total budgets (bits/image) for the outer-search approach.
                If you implement a direct minimal-bits LP/MILP instead, you may ignore this or use it as a coarse grid.

        Returns:
            int or None:
                Minimal B (bits/image) achieving ≥ α under your centralized method; or None if not achievable
                within the search/constraints you used.

        Notes:
            • This method does not prescribe a particular quantizer or classifier; it only requires that you
              respect train/val/test separation and report bits/image clearly.
            • If multiple solutions achieve α, return the smallest B according to your method.
        """
        # min_B = ...
        return min_B

    def minimal_bits_decentralized(self, feature_compressor, train_blocks, val_blocks, test_blocks, trainY, valY, testY, alpha, B_grid):
        """
        Task 3.3 (Decentralized)

        Goal:
            Given a target test accuracy α, find the minimal total bit budget B (bits/image) and a corresponding
            decentralized allocation (e.g., per-sensor parameters such as (b1, b2, b3, b4), if that matches your design)
            so that your decentralized formulation achieves test accuracy ≥ α.

        Allowed approaches (your choice, consistent with the guidelines):
            • Outer-search approach: use an outer search over candidate budgets and, for each budget,
              search allocations/solve your decentralized formulation on train/val and evaluate on test;
              return the smallest B achieving ≥ α and its chosen allocation.
              In this case, `B_grid` provides the candidate budgets you intend to try.
            • Direct optimization approach: encode the minimal-bits objective directly in an LP/ILP/MILP
              with decentralized constraints and accuracy ≥ α; solve for B and its allocation.
              In this case, `B_grid` may be ignored or used to warm-start/coarsely bracket solutions.

        Args:
            feature_compressor: an object exposing your decentralized pipeline (e.g., MyFeatureCompression) if you
                follow the outer-search route. For a direct LP/MILP approach, you may ignore it.
            train_blocks, val_blocks, test_blocks:
                lists of 4 arrays [N × d_s], one per sensor/quadrant; keep train/val/test strict.
            trainY, valY, testY: labels.
            alpha (float): target test accuracy in [0,1].
            B_grid (Iterable[int]): candidate total budgets (bits/image) for the outer-search approach.
                If you implement a direct minimal-bits LP/MILP instead, you may ignore this or use it as a scaffold.

        Returns:
            (int or None, tuple or None):
                (minimal B, a representation of the chosen allocation at that B) if achievable; otherwise (None, None).
                The “allocation” is whatever your decentralized design uses (e.g., (b1, b2, b3, b4) for scalar bit-depths,
                or any quantizer-specific parameterization you choose to report).

        Notes:
            • This method does not prescribe how you search or solve; it only requires that you respect
              train/val/test separation and clearly report bits/image and the corresponding allocation.
            • If multiple solutions achieve α, return the one with the smallest B according to your method.
        """
        # min_B, best_alloc = ..., ...
        return (min_B, best_alloc)
