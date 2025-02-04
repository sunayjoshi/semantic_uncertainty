"""Compute uncertainty measures after generating answers."""

from abc import abstractmethod
from typing import Callable, Union
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb
import matplotlib.pyplot as plt
import math
from functools import partial

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer

from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils


utils.setup_logger()

EXP_DETAILS = "experiment_details.pkl"


def check_and_reshape_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)  # Ensure X is an array
    if X.ndim == 0:
        return X.reshape(1, 1)  # Scalar -> (1,1)
    elif X.ndim == 1:
        return X.reshape(-1, 1)  # 1D array -> (n,1)
    return X  # Already 2D or higher


def generalized_inverse(
    y: Union[float, np.ndarray],
    theta: Callable,
    X: np.ndarray,
) -> np.ndarray:
    """
    Computes the generalized inverse of theta:
      g^{-1}(y) = inf { x in X : theta(x) <= y }

    Parameters:
        y: A float or an array-like of target values.
        theta: A callable function that maps x-values to predictions. theta should be a decreasing function.
        X: A sorted (monotonic) array of x values over which theta is defined.

    Returns:
        An array of x values corresponding to the inverse evaluation.
    """
    # make sure X and y are numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    # sort X in descending order
    X = np.sort(X)[::-1]
    # Evaluate decreasing theta function on the sorted x values. Results in an increasing array.
    theta_vals = theta(X)
    print("theta_vals", theta_vals)

    # Find the index where each y would be inserted to keep theta_vals sorted.
    # searchsorted assumes theta_vals is sorted in ascending order
    # "right" means that we choose the larger theta(X) value (smaller X value) in the case of a tie
    idx = np.searchsorted(theta_vals, y, side="right")
    print("idx where y would be inserted", idx)
    # searchsorted returns the index where y would be inserted to keep theta_vals sorted
    # so we need to subtract 1 to get the index of the largest theta(X) value that is less than or equal to y
    idx = idx - 1
    # We don't want to wrap around the end of the array, so we set any idx == -1 to 0
    idx = np.where(idx == -1, 0, idx)
    return X[idx]


class Recalibrator(TransformerMixin, BaseEstimator):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the recalibrator using uncertainty and accuracy values on some calibration set.

        Parameters:
            X (np.ndarray): The uncertainty values.
            y (np.ndarray): The accuracy values.
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Recalibrate uncertainty values.

        Parameters:
            X (np.ndarray): The uncertainty values to recalibrate.

        Returns:
            np.ndarray: The recalibrated uncertainty values.
        """
        ...


class IsotonicRecalibrator(Recalibrator):
    """
    Fits a nonincreasing isotonic regression (theta_star) and a local polynomial regression (r_hat_model),
    then returns a function that computes theta_star^{-1} ∘ r_hat_model.

    The inverse is defined as :math:`g^{-1}(y) = \sup{ x : g(x) >= y }`

    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the isotonic regression and piecewise-linear regression.

        Parameters:
            X (np.ndarray): The uncertainty values.
            y (np.ndarray): The accuracy values.
        """
        X = check_and_reshape_X(X)
        self.iso_reg = IsotonicRegression(increasing=False, out_of_bounds="clip")
        self.iso_reg.fit(X, y)

        # fit piecewise-constant, one-hot encoding the quantile bins
        self.piecewise_constant_regressor = Pipeline(
            [
                (
                    "binning",
                    KBinsDiscretizer(
                        n_bins=self.n_bins, encode="onehot", strategy="quantile"
                    ),
                ),
                ("regressor", LinearRegression(fit_intercept=True)),
            ]
        )
        self.piecewise_constant_regressor.fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Recalibrate uncertainty values.

        Parameters:
            X (np.ndarray): The uncertainty values to recalibrate.

        Returns:
            np.ndarray: The recalibrated uncertainty values.
        """
        X = check_and_reshape_X(X)
        r_hat = self.piecewise_constant_regressor.predict(X)
        theta_star = generalized_inverse(r_hat, self.iso_reg.predict, X)
        return theta_star


class LinearMonotonicRecalibrator(Recalibrator):
    """
    Fits a linear model from min to max, then uses the inverse of that model to recalibrate.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = check_and_reshape_X(X)
        self.linear_model = LinearRegression()
        # fit just a decreasing linear model from max to min
        self.linear_model.fit([np.min(X), np.max(X)], [np.max(y), np.min(y)])

        self.piecewise_constant_regressor = Pipeline(
            [
                (
                    "binning",
                    KBinsDiscretizer(
                        n_bins=self.n_bins, encode="onehot", strategy="quantile"
                    ),
                ),
                ("regressor", LinearRegression(fit_intercept=True)),
            ]
        )
        self.piecewise_constant_regressor.fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_and_reshape_X(X)
        r_hat = self.piecewise_constant_regressor.predict(X)
        theta_star = generalized_inverse(r_hat, self.linear_model.predict, X)
        return theta_star


# def isotonic_recalibrator(U, A):
#     """
#     Fits a nonincreasing isotonic regression (theta_star) and a local polynomial regression (r_hat_model),
#     then returns a function that computes theta_star^{-1} ∘ r_hat_model.

#     The inverse is defined as: g^{-1}(y) = sup{ x : g(x) >= y }

#     Parameters:
#     U (array-like): An array of n sampled uncertainties.
#     A (array-like): A corresponding array of n sampled accuracies.

#     Returns:
#     function: A function that takes uncertainty values and returns recalibrated values.
#     """

#     # Ensure U and A are numpy arrays
#     U = np.asarray(U)
#     A = np.asarray(A)

#     # Store original U mean/stddev for destandardization
#     U_mean = np.mean(U)
#     U_std = np.std(U)

#     # Store original A mean/stddev
#     A_mean = np.mean(A)
#     A_std = np.std(A)

#     # We can't fit regression functions if there's no variation in the independent variable
#     assert U_std != 0, "No variation in uncertainty values."
#     # Standardize U
#     U_standardized = (U - U_mean) / U_std

#     # Standardize A if A_std is nonzero
#     A_standardized = A
#     # if A_std != 0:
#     #     A_standardized = (A - A_mean) / A_std

#     # Sort standardized U and A
#     sorted_indices = np.argsort(U_standardized)
#     U_sorted = U_standardized[sorted_indices]
#     A_sorted = A_standardized[sorted_indices]

#     logging.info(f"[isotonic_recalibrator] Number of training points: {len(U_sorted)}")

#     # Fit theta_star: nonincreasing isotonic regression
#     iso_reg = IsotonicRegression(increasing=False, out_of_bounds="clip")
#     theta_star_values = iso_reg.fit_transform(U_sorted, A_sorted)

#     if np.sum(np.isnan(theta_star_values)) > 0:
#         raise ValueError("Isotonic regression returned NaN.")

#     ## Fit piecewise-constant regression, using cross-validation
#     plr_pipeline = Pipeline(
#         [
#             (
#                 "binning",
#                 KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile"),
#             ),
#             ("regressor", LinearRegression(fit_intercept=True)),
#         ]
#     )

#     param_grid = {
#         "binning__n_bins": [
#             5,
#             10,
#             15,
#             20,
#         ]  # Adjust based on data distribution and model complexity
#     }

#     search = GridSearchCV(
#         estimator=plr_pipeline,
#         param_grid=param_grid,
#         scoring="neg_mean_squared_error",  # 'r2'
#         cv=5,  # 5-fold cross-validation
#         n_jobs=-1,  # Use all available cores (optional)
#         verbose=0,  # Increase if you want more logs
#     )

#     search.fit(U_standardized.reshape(-1, 1), A_standardized)

#     # The best estimator from cross-validation:
#     r_hat_model = search.best_estimator_

#     logging.info(
#         f"Best piecewise-constant regression found with params: {search.best_params_}"
#     )

#     # Create plots directory if it doesn't exist
#     os.makedirs("./plots", exist_ok=True)

#     # Create plot
#     plt.figure(figsize=(10, 6))

#     # Plot original data points
#     plt.scatter(
#         U_standardized, A_standardized, alpha=0.3, label="Training Data", color="gray"
#     )

#     # Generate points for smooth curves
#     x_plot = np.linspace(U_standardized.min() - 1, U_standardized.max() + 1, 1000)

#     # Plot isotonic regression
#     y_iso = iso_reg.transform(x_plot)
#     plt.plot(x_plot, y_iso, "r-", label="Isotonic Regression", linewidth=2)

#     # Piecewise-constant logistic regression predictions
#     y_kernel = r_hat_model.predict(x_plot.reshape(-1, 1))
#     plt.step(
#         x_plot,
#         y_kernel,
#         where="mid",
#         label="Piecewise-Constant Regression",
#         linewidth=2,
#         color="b",
#     )

#     plt.xlabel("Standardized Uncertainty")
#     plt.ylabel("Accuracy")
#     plt.title("Isotonic and Regression Fits")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     # Save plot
#     plt.savefig("./plots/fitted_regression_curves.png", dpi=300, bbox_inches="tight")
#     plt.close()

#     logging.info("Saved calibration plot to ./plots/fitted_regression_curves.png")

#     # Create a second plot to visualize the density of standardized training uncertainties
#     plt.figure(figsize=(10, 6))
#     plt.hist(U_standardized, bins=30, density=True, alpha=0.7, color="blue")
#     plt.title("Density of Standardized Training Uncertainties")
#     plt.xlabel("Standardized Uncertainty")
#     plt.ylabel("Density")
#     plt.grid(True, alpha=0.3)

#     # Save the second plot
#     plt.savefig(
#         "./plots/standardized_uncertainty_density.png", dpi=300, bbox_inches="tight"
#     )
#     plt.close()

#     logging.info("Saved density plot to ./plots/standardized_uncertainty_density.png")

#     # def theta_star_inverse(y):
#     #     """
#     #     Computes the generalized inverse of theta_star:
#     #     g^{-1}(y) = sup{ x : g(x) >= y }

#     #     Parameters:
#     #     y (float or array-like): Value(s) to find the inverse for

#     #     Returns:
#     #     float or array: The inverse value(s)
#     #     """

#     #     y = np.asarray(y)
#     #     result = np.zeros_like(y)

#     #     for i, yi in enumerate(y.flat):
#     #         valid_points = theta_star_values >= yi
#     #         if not np.any(valid_points):
#     #             result.flat[i] = U_sorted[0]
#     #         else:
#     #             result.flat[i] = U_sorted[valid_points][-1]

#     #     return result.reshape(y.shape)

#     def theta_star_inverse(y):
#         """
#         Computes the generalized inverse of theta_star:
#         g^{-1}(y) = inf{ x : g(x) <= y }

#         Parameters:
#         y (float or array-like): Value(s) to find the inverse for

#         Returns:
#         float or array: The inverse value(s)
#         """

#         y = np.asarray(y)
#         result = np.zeros_like(y)

#         for i, yi in enumerate(y.flat):
#             valid_points = theta_star_values <= yi
#             if not np.any(valid_points):
#                 result.flat[i] = U_sorted[-1]  # Changed from first to last point
#             else:
#                 result.flat[i] = U_sorted[valid_points][
#                     0
#                 ]  # Changed from last to first valid point

#         return result.reshape(y.shape)

#     def recalibrator_function(u):
#         """
#         Applies the recalibration function theta_star^{-1} ∘ r_hat_model

#         Parameters:
#         u (float or array-like): Uncertainty value(s) to recalibrate

#         Returns:
#         float or array: Recalibrated uncertainty value(s)
#         """

#         if U_std == 0:
#             return u

#         u = np.asarray(u)
#         original_shape = u.shape

#         if u.ndim == 0:
#             u = u.reshape(1, 1)
#         elif u.ndim == 1:
#             u = u.reshape(-1, 1)

#         u_standardized = (u - U_mean) / U_std

#         # Regression predictions
#         pred_acc = r_hat_model.predict(u_standardized.reshape(-1, 1))

#         recalibrated_standardized = theta_star_inverse(pred_acc)
#         u_destandardized = recalibrated_standardized * U_std + U_mean

#         return u_destandardized.reshape(original_shape)

#     return recalibrator_function


def main(args):
    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.environ["USER"]
    scratch_dir = os.getenv("SCRATCH_DIR", ".")
    wandb_dir = f"{scratch_dir}/{user}/uncertainty"
    slurm_jobid = os.getenv("SLURM_JOB_ID", None)
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"
    if args.assign_new_wandb_id:
        logging.info("Assign new wandb_id.")
        api = wandb.Api()
        old_run = api.run(
            f"{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}"
        )
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f"slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}",
            # For convenience, keep any 'generate_answers' configs from old run,
            # but overwrite the rest!
            # NOTE: This means any special configs affecting this script must be
            # called again when calling this script!
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir
            )

            class Restored:
                name = f"{wandb.run.dir}/{filename}"

            return Restored
    else:
        logging.info("Reuse active wandb id.")

        def restore(filename):
            class Restored:
                name = f"{wandb.run.dir}/{filename}"

            return Restored

    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid,
            args.eval_wandb_runid,
        )

        is_ood_eval = True  # pylint: disable=invalid-name
        api = wandb.Api()
        old_run_train = api.run(
            f"{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}"
        )
        filename = "train_generations.pkl"
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir
        )
        with open(f"{wandb.run.dir}/{filename}", "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update(
            {"ood_training_set": old_run_train.config["dataset"]}, allow_val_change=True
        )
    else:
        is_ood_eval = False  # pylint: disable=invalid-name
        if args.compute_p_ik or args.compute_p_ik_answerable:
            train_generations_pickle = restore("train_generations.pkl")
            with open(train_generations_pickle.name, "rb") as infile:
                train_generations = pickle.load(infile)

    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info("Beginning loading for entailment model.")
        if args.entailment_model == "deberta":
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == "gpt-4":
            entailment_model = EntailmentGPT4(
                args.entailment_cache_id, args.entailment_cache_only
            )
        elif args.entailment_model == "gpt-3.5":
            entailment_model = EntailmentGPT35(
                args.entailment_cache_id, args.entailment_cache_only
            )
        elif args.entailment_model == "gpt-4-turbo":
            entailment_model = EntailmentGPT4Turbo(
                args.entailment_cache_id, args.entailment_cache_only
            )
        elif "llama" in args.entailment_model.lower():
            entailment_model = EntailmentLlama(
                args.entailment_cache_id,
                args.entailment_cache_only,
                args.entailment_model,
            )
        else:
            raise ValueError
        logging.info("Entailment model loading complete.")

    if args.compute_p_true_in_compute_stage:
        # This is usually not called.
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp["args"])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp["args"].dataset,
            add_options=old_exp["args"].use_mc_options,
            seed=args.random_seed,
        )
        del pt_validation_dataset

        # Reduce num generations used in p_true if needed!
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = (
            p_true_utils.construct_few_shot_prompt(
                model=pt_model,
                dataset=pt_train_dataset,
                indices=old_exp["p_true_indices"],
                prompt=old_exp["prompt"],
                brief=old_exp["BRIEF"],
                brief_always=old_exp["args"].brief_always
                and old_exp["args"].enable_brief,
                make_prompt=utils.get_make_prompt(old_exp["args"]),
                num_generations=num_gen,
                metric=utils.get_metric(old_exp["args"].metric),
            )
        )
        del p_true_responses
        wandb.config.update({"p_true_num_fewshot": len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info("Generated few-shot prompt for p_true.")
        logging.info(80 * "#")
        logging.info("p_true_few_shot_prompt: %s", p_true_few_shot_prompt)
        logging.info(80 * "#")

    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning(
            "Recompute accuracy enabled. This does not apply to precomputed p_true!"
        )
        metric = utils.get_metric(args.metric)

    # Restore outputs from `generate_answers.py` run.
    result_dict_pickle = restore("uncertainty_measures.pkl")
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict["semantic_ids"] = []

    validation_generations_pickle = restore("validation_generations.pkl")
    with open(validation_generations_pickle.name, "rb") as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation["reference"]["answers"]["text"]) > 0

    # Store training entropies and accuracies, and fit the recalibrator.
    if args.compute_recalibrated_entropy:
        train_semantic_entropies = []
        train_accuracies = []

        # First pass through training data to collect entropies.
        for tid in train_generations:
            example = train_generations[tid]
            full_responses = example["responses"]

            if not args.use_all_generations:
                responses = [fr[0] for fr in full_responses[: args.use_num_generations]]
                log_liks = [r[1] for r in full_responses[: args.use_num_generations]]
            else:
                responses = [fr[0] for fr in full_responses]
                log_liks = [r[1] for r in full_responses]

            semantic_ids = get_semantic_ids(
                responses,
                model=entailment_model,
                strict_entailment=args.strict_entailment,
                example=example,
            )

            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
            log_likelihood_per_semantic_id = logsumexp_by_id(
                semantic_ids, log_liks_agg, agg="sum_normalized"
            )
            entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)

            logging.info(f"Training entropy: {entropy}")

            train_semantic_entropies.append(entropy)
            train_accuracies.append(example["most_likely_answer"]["accuracy"])

        entropy_std = np.std(np.asarray(train_semantic_entropies))
        logging.info(f"Stddev of training entropies: {entropy_std}")

        logging.info(
            f"Type of train_semantic_entropies: {type(train_semantic_entropies)}"
        )
        logging.info(
            f"Shape of train_semantic_entropies: {np.asarray(train_semantic_entropies).shape}"
        )
        logging.info(f"Type of train_accuracies: {type(train_accuracies)}")
        logging.info(f"Shape of train_accuracies: {np.asarray(train_accuracies).shape}")

        # Fit the recalibrator.
        # recalibrator = isotonic_recalibrator(
        #     np.array(train_semantic_entropies), np.array(train_accuracies)
        # )
        recalibrator = IsotonicRecalibrator()
        recalibrator.fit(np.array(train_semantic_entropies), np.array(train_accuracies))

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        question = example["question"]
        context = example["context"]
        full_responses = example["responses"]
        most_likely_answer = example["most_likely_answer"]

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[: args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.recompute_accuracy:
            logging.info("Recomputing accuracy!")
            if is_answerable(example):
                acc = metric(most_likely_answer["response"], example, None)
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_is_true.append(acc)
            logging.info("Recomputed accuracy!")

        else:
            validation_is_true.append(most_likely_answer["accuracy"])

        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer["embedding"])
        logging.info("validation_is_true: %f", validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[: args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies["context_entails_response"].append(
                    context_entails_response(context, responses, entailment_model)
                )

            if args.condition_on_question and args.entailment_model == "deberta":
                responses = [f"{question} {r}" for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses,
                model=entailment_model,
                strict_entailment=args.strict_entailment,
                example=example,
            )

            result_dict["semantic_ids"].append(semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            entropies["cluster_assignment_entropy"].append(
                cluster_assignment_entropy(semantic_ids)
            )

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute naive entropy.
            entropies["regular_entropy"].append(predictive_entropy(log_liks_agg))

            # Compute semantic entropy.
            log_likelihood_per_semantic_id = logsumexp_by_id(
                semantic_ids, log_liks_agg, agg="sum_normalized"
            )
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies["semantic_entropy"].append(pe)

            # Compute recalibrated entropy.
            if args.compute_recalibrated_entropy:
                logging.info(f"Type of pe: {type(pe)}")
                # recalibrated_pe = recalibrator(pe).item()
                recalibrated_pe = recalibrator.transform(pe)
                logging.info(f"Type of recalibrated_pe: {type(recalibrated_pe)}")
                entropies["recalibrated_semantic_entropy"].append(recalibrated_pe)

            # pylint: disable=invalid-name
            log_str = "semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s"
            entropies_fmt = ", ".join(
                [f"{i}:{j[-1]:.2f}" for i, j in entropies.items()]
            )
            # pylint: enable=invalid-name
            logging.info(80 * "#")
            logging.info("NEW ITEM %d at id=`%s`.", idx, tid)
            logging.info("Context:")
            logging.info(example["context"])
            logging.info("Question:")
            logging.info(question)
            logging.info("True Answers:")
            logging.info(example["reference"])
            logging.info("Low Temperature Generation:")
            logging.info(most_likely_answer["response"])
            logging.info("Low Temperature Generation Accuracy:")
            logging.info(most_likely_answer["accuracy"])
            logging.info("High Temp Generation:")
            logging.info([r[0] for r in full_responses])
            logging.info("High Temp Generation:")
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model,
                question,
                most_likely_answer["response"],
                responses,
                p_true_few_shot_prompt,
                hint=old_exp["args"].p_true_hint,
            )
            p_trues.append(p_true)
            logging.info("p_true: %s", np.exp(p_true))

        count += 1
        if count >= args.num_eval_samples:
            logging.info("Breaking out of main loop.")
            break

    logging.info("Accuracy on original task: %f", np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict["validation_is_false"] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict["validation_unanswerable"] = validation_unanswerable
    logging.info(
        "Unanswerable prop on validation: %f", np.mean(validation_unanswerable)
    )

    if "uncertainty_measures" not in result_dict:
        result_dict["uncertainty_measures"] = dict()

    if args.compute_predictive_entropy:
        result_dict["uncertainty_measures"].update(entropies)

    if args.compute_p_ik or args.compute_p_ik_answerable:
        # Assemble training data for embedding classification.
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]["most_likely_answer"]
            train_embeddings.append(most_likely_answer["embedding"])
            train_is_true.append(most_likely_answer["accuracy"])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info(
            "Unanswerable prop on p_ik training: %f", np.mean(train_unanswerable)
        )

    if args.compute_p_ik:
        logging.info("Starting training p_ik on train embeddings.")
        # Train classifier of correct/incorrect from embeddings.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings,
            is_false=train_is_false,
            eval_embeddings=validation_embeddings,
            eval_is_false=validation_is_false,
        )
        result_dict["uncertainty_measures"]["p_ik"] = p_ik_predictions
        logging.info("Finished training p_ik on train embeddings.")

    if args.compute_p_ik_answerable:
        # Train classifier of answerable/unanswerable.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings,
            is_false=train_unanswerable,
            eval_embeddings=validation_embeddings,
            eval_is_false=validation_unanswerable,
        )
        result_dict["uncertainty_measures"]["p_ik_unanswerable"] = p_ik_predictions

    if args.compute_p_true_in_compute_stage:
        result_dict["uncertainty_measures"]["p_false"] = [1 - p for p in p_trues]
        result_dict["uncertainty_measures"]["p_false_fixed"] = [
            1 - np.exp(p) for p in p_trues
        ]

    utils.save(result_dict, "uncertainty_measures.pkl")

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
        logging.info(50 * "#X")
        logging.info("STARTING `analyze_run`!")
        analyze_run(wandb.run.id)
        logging.info(50 * "#X")
        logging.info("FINISHED `analyze_run`!")


if __name__ == "__main__":
    parser = utils.get_parser(stages=["compute"])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f"Unkown args: {unknown}")

    logging.info("Args: %s", args)

    main(args)
