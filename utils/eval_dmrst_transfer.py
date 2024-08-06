import os
from glob import glob

import fire
import numpy as np

from utils.eval_dmrst import EvalDMRST


class EvalDMRSTTransfer:
    """
    Evaluates transfer learning performance for DMRST models.

    Args:
        models_dir (str): Directory containing model checkpoints.
        nfolds (int): Number of cross-validation folds.
        corpus (str): Corpus name (e.g., 'GUM').
        lang (str): Language code (e.g., 'ru').
        cuda_device (int, optional): CUDA device ID (default is 0).

    Attributes:
        models_dir (str): Directory containing model checkpoints.
        nfolds (int): Number of cross-validation folds.
        corpus (str): Corpus name.
        lang (str): Language code.
        cuda_device (int): CUDA device ID.

    Methods:
        evaluate(): Computes evaluation metrics for each genre and overall.

    Example:
        To evaluate transfer performance:
        ```
        python utils/eval_dmrst_transfer.py --models_dir saves/rrg_+tony --corpus 'GUM' --lang 'ru' --nfolds 5 evaluate
        ```

    """

    def __init__(self, models_dir: str, nfolds: int, corpus: str, lang: str, cuda_device=0):
        self.models_dir = models_dir
        self.nfolds = nfolds
        self.corpus = corpus
        self.lang = lang
        self.cuda_device = cuda_device

    def evaluate(self):
        """
        Computes evaluation metrics for each genre and overall.

        Returns:
            dict: Dictionary containing genre-wise and overall metrics.
        """

        all_metrics = []
        for path in sorted(glob(os.path.join(self.models_dir, '*')))[:self.nfolds]:
            evaluator = EvalDMRST(path, self.corpus, self.lang, cuda_device=self.cuda_device)
            metrics = evaluator.by_genre()
            metrics['all'] = evaluator.full()
            all_metrics.append(metrics)

        metrics_stats = dict()
        for genre in all_metrics[0].keys():
            metrics_stats[genre] = dict()
            for key in all_metrics[0][genre].keys():
                metrics_stats[genre][key] = (np.mean([metric[genre][key] for metric in all_metrics]),
                                             np.std([metric[genre][key] for metric in all_metrics]))

        return metrics_stats


if __name__ == '__main__':
    fire.Fire(EvalDMRSTTransfer)
