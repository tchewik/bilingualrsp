import os
import pickle

import pandas as pd
from tqdm import tqdm

from relation_classifier.feature_rich_clf import FeatureRichClassifier
from utils.du_converter import DUConverter


class Reclassifier:
    def __init__(self, predictions_path, filenames, annotations_path, corpus):
        self.trees = DUConverter(predictions_path).data
        self.filenames = filenames
        self.annotations_path = annotations_path

        self.fr_classifier = FeatureRichClassifier(corpus=corpus, lang='ru')

    @staticmethod
    def _current_pair(du):
        """
        Collects information about all the relations in the RST tree.

        Args:
            du: RST tree or its non-elementary DU.

        Returns:
            List of tuples containing basic fields from the feature table.
        """

        if du.relation == 'elementary':
            return []

        return [(du.left.text, du.right.text, du.left.start, du.right.start, du.relation, du.nuclearity)
                ] + Reclassifier._current_pair(du.left) + Reclassifier._current_pair(du.right)

    def _construct_df(self):
        """
        Constructs the pandas dataframe with all the features.

        Returns:
            Pandas DataFrame.
        """

        from relation_classifier.feature_processors import FeaturesProcessor
        fp = FeaturesProcessor(language='ru', verbose=0, use_use=True, use_sentiment=True)

        all_data = []
        for tree, filename in tqdm(zip(self.trees, self.filenames), total=len(self.filenames)):
            df = pd.DataFrame(self._current_pair(tree), columns=['snippet_x', 'snippet_y',
                                                                 'loc_x', 'loc_y',
                                                                 'pred_rel', 'pred_nuc'])
            annot = pickle.load(open(os.path.join(self.annotations_path, filename + '.pkl'), 'rb'))
            features = fp(df,
                          annot['text'], annot['tokens'],
                          annot['sentences'], annot['lemma'],
                          annot['morph'], annot['postag'],
                          annot['syntax_dep_tree'], )

            features['filename'] = filename
            all_data.append(features)

        self.all_data = pd.concat(all_data)

    def _scale_data(self):
        self.fr_classifier.load_the_save()

        drop_columns = self.fr_classifier.drop_columns + ['pred_rel', 'pred_nuc']
        drop_columns = [col for col in drop_columns if col in self.all_data.columns]
        x = self.all_data.drop(columns=drop_columns)

        scaled_np = self.fr_classifier.scaler.transform(x)
        x = pd.DataFrame(scaled_np, index=self.all_data.index)

        return x

    def predict_fr(self):
        return self.fr_classifier._make_predictions(self._scale_data())
