"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb
import matplotlib.pyplot as plt
import math 

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

EXP_DETAILS = 'experiment_details.pkl'


def isotonic_recalibrator(U, A):
    """
    Fits a single decreasing line and a piecewise-constant regression (r_hat_model),
    then returns a function that computes that line's inverse ∘ r_hat_model.

    The inverse is defined as: g^{-1}(y) = (y - intercept) / slope, 
    clipped to the [min(U), max(U)] range.
    """
    
    # Ensure U and A are numpy arrays
    U = np.asarray(U)
    A = np.asarray(A)

    logging.info(f"[isotonic_recalibrator] Number of training points: {len(U)}")
    
    # Fit a decreasing line
    x_min = np.min(U)
    x_max = np.max(U)
    y_max = np.max(A)
    y_min = np.min(A)
    
    slope = (y_min - y_max) / (x_max - x_min)
    intercept = y_max - slope * x_min
    
    def linear_theta_star(x):
        return slope * x + intercept

    # Inverse of theta
    def theta_star_inverse(y):
        x = (y - intercept) / slope
        # Clip to [x_min, x_max]
        x = np.clip(x, x_min, x_max)
        return x

    # Piecewise-constant regression with 10 bins 
    plr_pipeline = Pipeline([
        ('binning', KBinsDiscretizer(n_bins=7, encode='onehot', strategy='uniform')), # 10, 'quantile'
        ('regressor', LinearRegression(fit_intercept=True))
    ])

    plr_pipeline.fit(U.reshape(-1, 1), A)
    r_hat_model = plr_pipeline

    # Plots 
    os.makedirs('./plots', exist_ok=True)
    
    # Plot the fitted line 
    plt.figure(figsize=(10, 6))
    x_plot = np.linspace(U.min() - 1, U.max() + 1, 1000)
    y_line = linear_theta_star(x_plot)
    plt.plot(x_plot, y_line, 'r-', label='Decreasing Line', linewidth=2)
    
    # Plot piecewise-constant regression
    y_kernel = r_hat_model.predict(x_plot.reshape(-1, 1))
    plt.step(x_plot, y_kernel, where='mid', label='Piecewise-Constant Regression', linewidth=2, color='b')
    
    plt.xlabel('Uncertainty')
    plt.ylabel('Accuracy')
    plt.title('Decreasing Line + Piecewise-Constant Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compute binned average accuracy using equispaced bins
    num_bins = 10
    bin_edges = np.linspace(U.min(), U.max(), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_acc = []

    for i in range(num_bins):
        # For the last bin include the right edge to capture all values
        if i == num_bins - 1:
            mask = (U >= bin_edges[i]) & (U <= bin_edges[i + 1])
        else:
            mask = (U >= bin_edges[i]) & (U < bin_edges[i + 1])
            
        if np.any(mask):
            binned_acc.append(np.mean(A[mask]))
        else:
            binned_acc.append(np.nan)  # Handle empty bins if any

    # Plot the binned estimate
    plt.plot(bin_centers, binned_acc, 'o-', label='Binned Estimate (Equispaced)', color='green', linewidth=2)

    plt.savefig('./plots/fitted_regression_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info('Saved calibration plot to ./plots/fitted_regression_curves.png')

    # Plot histogram of training U 
    plt.figure(figsize=(10, 6))
    plt.hist(U, bins=30, density=False, alpha=0.7, color='blue')
    plt.title('Histogram of Training Uncertainties')
    plt.xlabel('Uncertainty')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('./plots/uncertainty_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info('Saved histogram to ./plots/uncertainty_histogram.png')
    
    def recalibrator_function(u):
        """
        Applies the recalibration function line^{-1} ∘ r_hat_model
        """

        u = np.asarray(u)
        original_shape = u.shape
        
        if u.ndim == 0:
            u = u.reshape(1, 1)
        elif u.ndim == 1:
            u = u.reshape(-1, 1)
                
        # Predicted "accuracy"
        pred_acc = r_hat_model.predict(u)
        
        # Use the line-based inverse
        u_recalibrated = theta_star_inverse(pred_acc)
        
        return u_recalibrated.reshape(original_shape)
    
    return recalibrator_function


def main(args):

    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.environ['USER']
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"
    if args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
            # For convenience, keep any 'generate_answers' configs from old run,
            # but overwrite the rest!
            # NOTE: This means any special configs affecting this script must be
            # called again when calling this script!
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored
    else:
        logging.info('Reuse active wandb id.')

        def restore(filename):
            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored

    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid, args.eval_wandb_runid)

        is_ood_eval = True  # pylint: disable=invalid-name
        api = wandb.Api()
        old_run_train = api.run(f'{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}')
        filename = 'train_generations.pkl'
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir)
        with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update(
            {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    else:
        is_ood_eval = False  # pylint: disable=invalid-name
        if args.compute_p_ik or args.compute_p_ik_answerable:
            train_generations_pickle = restore('train_generations.pkl')
            with open(train_generations_pickle.name, 'rb') as infile:
                train_generations = pickle.load(infile)

    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')

    if args.compute_p_true_in_compute_stage:
        # This is usually not called.
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp['args'].dataset, add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed)
        del pt_validation_dataset

        # Reduce num generations used in p_true if needed!
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=utils.get_metric(old_exp['args'].metric))
        del p_true_responses
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info('Generated few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning('Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = utils.get_metric(args.metric)

    # Restore outputs from `generate_answers.py` run.
    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict['semantic_ids'] = []

    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    # Store training entropies and accuracies, and fit the recalibrator.
    if args.compute_recalibrated_entropy:
        train_semantic_entropies = []
        train_accuracies = []
        
        # First pass through training data to collect entropies.
        for tid in train_generations:
            example = train_generations[tid]
            full_responses = example["responses"]
            
            if not args.use_all_generations:
                responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                responses = [fr[0] for fr in full_responses]
                log_liks = [r[1] for r in full_responses]
                
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)
                
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)

            logging.info(f"Training entropy: {entropy}")
            
            train_semantic_entropies.append(entropy)
            train_accuracies.append(example['most_likely_answer']['accuracy'])
        
        entropy_std = np.std(np.asarray(train_semantic_entropies))
        logging.info(f"Stddev of training entropies: {entropy_std}")

        logging.info(f"Type of train_semantic_entropies: {type(train_semantic_entropies)}")
        logging.info(f"Shape of train_semantic_entropies: {np.asarray(train_semantic_entropies).shape}")
        logging.info(f"Type of train_accuracies: {type(train_accuracies)}")
        logging.info(f"Shape of train_accuracies: {np.asarray(train_accuracies).shape}")

        # Fit the recalibrator.
        recalibrator = isotonic_recalibrator(
            np.array(train_semantic_entropies), 
            np.array(train_accuracies)
        )

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(validation_generations):

        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if is_answerable(example):
                acc = metric(most_likely_answer['response'], example, None)
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')

        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        logging.info('validation_is_true: %f', validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)

            result_dict['semantic_ids'].append(semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute naive entropy.
            entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))

            # Compute semantic entropy.
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            # Compute recalibrated entropy.
            if args.compute_recalibrated_entropy:
                logging.info(f'Type of pe: {type(pe)}')
                recalibrated_pe = recalibrator(pe).item()
                logging.info(f'Type of recalibrated_pe: {type(recalibrated_pe)}')
                entropies['recalibrated_semantic_entropy'].append(recalibrated_pe)

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', np.exp(p_true))

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    if args.compute_p_ik or args.compute_p_ik_answerable:
        # Assemble training data for embedding classification.
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    if args.compute_p_ik:
        logging.info('Starting training p_ik on train embeddings.')
        # Train classifier of correct/incorrect from embeddings.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
        logging.info('Finished training p_ik on train embeddings.')

    if args.compute_p_ik_answerable:
        # Train classifier of answerable/unanswerable.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    utils.save(result_dict, 'uncertainty_measures.pkl')

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        analyze_run(wandb.run.id)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
