import pickle

import fire
import numpy as np
import torch
from tqdm import tqdm

from dmrst_parser.data_manager import DataManager
from dmrst_parser.predictor import Predictor
from dmrst_parser.src.parser.metrics import get_batch_metrics
from dmrst_parser.src.parser.metrics import get_micro_metrics
from utils.corpus_stats import select_genre


class EvalDMRST:
    """
    This class evaluates the performance of a DMRST parser model on a given test corpus.

    Args:
        model_path (str): The path to the trained model.
        corpus (str): The name of the corpus to evaluate the model on.
        lang (str): The language of the corpus.
        cuda_device (int): The ID of the CUDA device to use for evaluation.

    Attributes:
        model_path (str): The path to the trained model.
        corpus (str): The name of the corpus to evaluate the model on.
        predictor (Predictor): The predictor object used for parsing.
        data (Data): The test data to evaluate the model on.
        filenames (list): The filenames of the test data.

    Methods:
        eval_model: Evaluates the performance of the model on the test data.
        save_trees: Runs the predictor on given data and saves the predictions.

    Example:
        To evaluate model performance:
        ```
        python utils/eval_dmrst.py --model_path saves/model_path \
                                   --corpus 'GUM' \
                                   --lang 'ru' \
                                   --cuda_device 0 \
                                   by_genre
        ```
        To obtain the predictions:
        ```
        python utils/eval_dmrst.py --model_path saves/model_path \
                                   --corpus 'GUM' \
                                   --lang 'ru' \
                                   --cuda_device 0 \
                                   save_trees \
                                   --output_path saves/model_path/test_predictions.pkl
        ```
    """

    def __init__(self, model_path, corpus, lang, cuda_device=0):
        """
        Initializes the EvalDMRST object.

        Args:
            model_path (str): The path to the trained model.
            corpus (str): The name of the corpus to evaluate the model on.
            lang (str): The language of the corpus.
            cuda_device (int): The ID of the CUDA device to use for evaluation.
        """

        self.model_path = model_path
        self.corpus = corpus

        self.predictor = Predictor(model_path, cuda_device=cuda_device)

        dm_file = f'data/data_manager_{corpus.lower()}.pickle'
        dm = DataManager(corpus=corpus).from_pickle(dm_file)
        self.data = dm.get_data(lang=lang)[-1]  # test data
        self.data = self.predictor.tokenize(self.data)
        self.filenames = dm.corpus['test']

    def eval_model(self, data, use_pred_segmentation=True, use_org_parseval=True, batch_size=100, return_trees=False):
        """
        Evaluates the performance of the model on the given test data.

        Args:
            data (Data): The test data to evaluate the model on.
            use_pred_segmentation (bool): Whether to use predicted segmentation.
            use_org_parseval (bool): Whether to use original parseval.
            batch_size (int): The batch size to use for evaluation.
            return_trees (bool): Whether to return the parse trees.

        Returns:
            dict: A dictionary containing the evaluation metrics.
            if return_trees:
            Data: Predictions.
        """

        loss_tree_all = []
        loss_label_all = []
        correct_span = 0
        correct_relation = 0
        correct_nuclearity = 0
        correct_full = 0
        no_system = 0
        no_golden = 0
        no_gold_seg = 0
        no_pred_seg = 0
        no_correct_seg = 0

        # Macro
        correct_span_list = []
        correct_relation_list = []
        correct_nuclearity_list = []
        correct_full_list = []
        no_system_list = []
        no_golden_list = []

        if return_trees:
            predictions = {
                'tokens': [],
                'spans': [],
                'edu_breaks': [],
                'true_spans': [],
                'true_edu_breaks': []
            }

        batches = self.predictor.get_batches(data, batch_size)
        pbar = tqdm(enumerate(batches), total=len(batches), leave=False)
        for i, batch in pbar:
            with torch.no_grad():
                loss_tree_batch, loss_label_batch, \
                    span_batch, label_tuple_batch, predict_edu_breaks = self.predictor.model.testing_loss(
                    batch.input_sentences, batch.sent_breaks, batch.entity_ids, batch.entity_position_ids,
                    batch.edu_breaks, batch.relation_label, batch.parsing_breaks,
                    generate_tree=True, use_pred_segmentation=use_pred_segmentation)

            if return_trees:
                predictions['tokens'] += [self.predictor.tokenizer.convert_ids_to_tokens(text) for text in
                                          batch.input_sentences]
                predictions['spans'] += span_batch
                predictions['edu_breaks'] += predict_edu_breaks
                predictions['true_spans'] += batch.golden_metric
                predictions['true_edu_breaks'] += batch.edu_breaks

            metrics = get_batch_metrics(pred_spans_batch=span_batch, gold_spans_batch=batch.golden_metric,
                                        pred_edu_breaks_batch=predict_edu_breaks,
                                        gold_edu_breaks_batch=batch.edu_breaks,
                                        use_org_parseval=use_org_parseval)

            (correct_span_batch, correct_relation_batch, correct_nuclearity_batch, correct_full_batch,
             no_system_batch, no_golden_batch, correct_span_batch_list, correct_relation_batch_list,
             correct_nuclearity_batch_list, correct_full_batch_list, no_system_batch_list, no_golden_batch_list,
             segment_results_list) = metrics

            loss_tree_all.append(loss_tree_batch)
            loss_label_all.append(loss_label_batch)

            correct_span += correct_span_batch
            correct_relation += correct_relation_batch
            correct_nuclearity += correct_nuclearity_batch
            correct_full += correct_full_batch
            no_system += no_system_batch
            no_golden += no_golden_batch
            no_gold_seg += segment_results_list[0]
            no_pred_seg += segment_results_list[1]
            no_correct_seg += segment_results_list[2]

            correct_span_list += correct_span_batch_list
            correct_nuclearity_list += correct_nuclearity_batch_list
            correct_relation_list += correct_relation_batch_list
            correct_full_list += correct_full_batch_list

            no_system_list += no_system_batch_list
            no_golden_list += no_golden_batch_list

        span_points, relation_points, nuclearity_points, f1_full, segment_points = get_micro_metrics(correct_span,
                                                                                                     correct_relation,
                                                                                                     correct_nuclearity,
                                                                                                     correct_full,
                                                                                                     no_system,
                                                                                                     no_golden,
                                                                                                     no_gold_seg,
                                                                                                     no_pred_seg,
                                                                                                     no_correct_seg)

        seg_pr, seg_re, seg_f1 = segment_points
        span_pr, span_re, span_f1 = span_points
        nuc_pr, nuc_re, nuc_f1 = nuclearity_points
        rel_pr, rel_re, rel_f1 = relation_points

        metrics = {
            'f1_seg': seg_f1,
            'f1_span': span_f1,
            'f1_nuclearity': nuc_f1,
            'f1_relation': rel_f1,
            'f1_full': f1_full,
        }

        if return_trees:
            return metrics, predictions

        return metrics

    def print_metrics(self, metrics, genre='all'):
        """
        Prints evaluation metrics in a latex-friendly &-separated format.

        Args:
            metrics (dict): Dictionary containing evaluation metrics.
            genre (str): Genre name (default is 'all').
        """

        print(genre, end=' \t& ')
        for metric in metrics:
            if metric.startswith('f1_'):
                print(np.round(metrics[metric] * 100, 2), end='  & ')

        print()

    def full(self, return_trees=False):
        """
        Computes and prints overall evaluation metrics.

        Args:
            return_trees (bool): Whether to return parsed trees (default is False).

        Returns:
            dict or tuple: Overall evaluation metrics or metrics along with parsed trees.
        """

        if return_trees:
            metrics, trees = self.eval_model(self.data, return_trees=True)
            return metrics, trees

        else:
            metrics = self.eval_model(self.data)
            self.print_metrics(metrics)
            return metrics

    def by_genre(self):
        """
        Computes and returns evaluation metrics by genre.

        Returns:
            dict: Dictionary containing genre-wise evaluation metrics.
        """
        if self.corpus == 'GUM':
            genres = ['academic', 'bio', 'conversation', 'fiction', 'interview',
                      'news', 'reddit', 'speech', 'textbook', 'vlog', 'voyage', 'whow']

        if self.corpus == 'RuRSTB':
            genres = ['news', 'blogs']

        results = dict()
        for genre in genres:
            cd, _ = select_genre(self.data, self.filenames, genre)
            metrics = self.eval_model(cd)
            self.print_metrics(metrics, genre)

            results[genre] = metrics.copy()

        return results

    def save_trees(self, output_path):
        """
        Saves the predictions in the given pickle file.

        Args:
            output_path (str): Path to the output pickle file.
        """
        metrics, trees = self.eval_model(self.data, return_trees=True)
        with open(output_path, 'wb') as f:
            pickle.dump(trees, f)


if __name__ == '__main__':
    fire.Fire(EvalDMRST)
