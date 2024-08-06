import os
import pickle
from pathlib import Path

import fire
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d
from tqdm.autonotebook import tqdm

from dmrst_parser.data_manager import DataManager


class MyLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y)
        return self


class FeatureRichClassifier:
    def __init__(self, corpus, lang, host_spacy='localhost', port_spacy='3334'):
        self.corpus = corpus
        self.lang = lang
        self.host_spacy = host_spacy
        self.port_spacy = port_spacy

        self.data_path = Path(os.path.join('data', 'relation_pairs'))

        self.annotation_path = Path(os.path.join('data', 'annot_dumps', self.corpus, self.lang))
        self.annotation_path.mkdir(parents=True, exist_ok=True)
        self.features_path = Path(os.path.join('data', 'feature_dumps', self.corpus, self.lang))
        self.features_path.mkdir(parents=True, exist_ok=True)

        corpus_dump = os.path.join('data', f'data_manager_{corpus.lower()}.pickle')
        self.dm = DataManager(corpus=corpus).from_pickle(corpus_dump)

        self.random_state = 45
        self.scaler = None
        self.drop_columns = None
        self.model = None
        self.label_encoder = None

        self.save_path = Path(os.path.join('saves', f'frc_{self.corpus}_{self.lang}.pkl'))

    def _load_pairs(self):
        data = []
        for part in ('train', 'dev', 'test'):
            path = os.path.join('data', 'relation_pairs', f'{self.corpus}_{self.lang}_{part}.fth')
            current = pd.read_feather(path)
            current['rel'] = current.category_id + '_' + current.order

            data.append(current)

        return data

    def make_annotations(self):
        from isanlp import PipelineCommon
        from isanlp.processor_razdel import ProcessorRazdel
        from isanlp.processor_remote import ProcessorRemote

        ppl = PipelineCommon([
            (ProcessorRazdel(), ['text'],
             {'tokens': 'tokens',
              'sentences': 'sentences'}),
            (ProcessorRemote(self.host_spacy, self.port_spacy, '0'),
             ['tokens', 'sentences'],
             {'lemma': 'lemma',
              'postag': 'postag',
              'morph': 'morph',
              'syntax_dep_tree': 'syntax_dep_tree',
              'entities': 'entities'})
        ])

        filenames = [self.dm.corpus.get(part) for part in ('train', 'dev', 'test')]
        for fn, data_part in zip(filenames, self.dm.get_data(lang=self.lang)):
            for filename, tokens in tqdm(zip(fn, data_part.input_sentences), total=len(fn)):
                text = ' '.join(tokens)
                annot = ppl(text)
                pickle.dump(annot, open(self.annotation_path.joinpath(f'{filename}.pkl'), 'wb'))

    def extract_features(self):
        train, dev, test = self._load_pairs()

        from relation_classifier.feature_processors import FeaturesProcessor
        fp = FeaturesProcessor(language=self.lang, verbose=0, use_use=True, use_sentiment=True)

        table = pd.concat([train, dev, test])
        for filename, df in tqdm(table.groupby('filename')):
            annot = pickle.load(open(self.annotation_path.joinpath(f'{filename}' + '.pkl'), 'rb'))
            features = fp(df,
                          annot['text'], annot['tokens'],
                          annot['sentences'], annot['lemma'],
                          annot['morph'], annot['postag'],
                          annot['syntax_dep_tree'], )

            features.to_pickle(self.features_path.joinpath(filename + '.feats.pkl'))

    def load_data(self):
        self.data = {part: None for part in ('train', 'dev', 'test')}
        for part in self.data.keys():
            tables = []
            for filename in self.dm.corpus.get(part):
                tables.append(pd.read_pickle(self.features_path.joinpath(filename + '.feats.pkl')))

            self.data[part] = pd.concat(tables).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

    def collect_constants(self):
        df = pd.concat([self.data['train'], self.data['dev'], self.data['test']])
        df = df.fillna(0.)

        constants = [c for c in df.drop(columns=['snippet_x_tokens', 'snippet_y_tokens']).columns if
                     len(set(df[c])) == 1]
        to_drop = ['index', 'snippet_x', 'snippet_y', 'snippet_x_tokens', 'snippet_y_tokens', 'filename', 'order']

        self.drop_columns = constants + to_drop

    def _prepare_model(self):
        counts = self.data['train']['rel'].value_counts(normalize=False).values
        catboost = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            random_seed=self.random_state,
            verbose=2,
            loss_function='MultiClass',
            class_weights=counts / counts[-1],
            eval_metric='TotalF1',
            task_type="GPU",
            devices='0'
        )

        fs_catboost = Pipeline([
            ('feature_selection',
             SelectFromModel(LogisticRegression(penalty='l1', solver='saga', max_iter=200, C=1., n_jobs=-1))),
            ('classification', catboost),
        ], verbose=True)

        logreg = LogisticRegression(random_state=self.random_state,
                                    solver='lbfgs',
                                    n_jobs=-1,
                                    C=0.002,
                                    multi_class='multinomial',
                                    class_weight='balanced')

        fs_catboost_plus_logreg = VotingClassifier(
            [('fs_catboost', fs_catboost), ('logreg', logreg)], voting='soft', n_jobs=-1)

        return fs_catboost_plus_logreg

    def _scale_data(self, part, retrain=False):
        if part == 'train':
            y, x = (self.data['train']['rel'].to_frame(),
                    self.data['train'].drop('category_id', axis=1).drop(columns=self.drop_columns + ['rel']))
        elif part == 'dev':
            y, x = (self.data['dev']['rel'].to_frame(),
                    self.data['dev'].drop('category_id', axis=1).drop(columns=self.drop_columns + ['rel']))
        elif part == 'test':
            y, x = (self.data['test']['rel'].to_frame(),
                    self.data['test'].drop('category_id', axis=1).drop(columns=self.drop_columns + ['rel']))

        if retrain and part == 'train':
            self.scaler = StandardScaler().fit(x)

        scaled_np = self.scaler.transform(x)
        x = pd.DataFrame(scaled_np, index=self.data[part].index)

        return x, y

    def _only_scale(self, data):
        y, x = (data['rel'].to_frame(),
                data.drop('category_id', axis=1).drop(columns=self.drop_columns + ['rel']))

        scaled_np = self.scaler.transform(x)
        x = pd.DataFrame(scaled_np, index=data.index)

        return x, y

    def train(self, extract_features=False):
        if extract_features:
            self.extract_features()

        self.load_data()
        self.collect_constants()

        X_train, y_train, = self._scale_data(part='train', retrain=True)

        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y_train)

        self.model = self._prepare_model()
        self.model.fit(X_train, y_train)

        pickle.dump(self, open(self.save_path, 'wb'))

    def _make_predictions(self, scaled_data):
        probs = self.model.predict_proba(scaled_data)
        return [dict(zip(self.label_encoder.classes_, pred)) for pred in probs]

    def predict(self, part: str):
        x, y = self._scale_data(part)
        return self._make_predictions(x)

    def predict_on_other(self, corpus, lang, dict_only=False):
        other_clf = FeatureRichClassifier(corpus, lang)
        other_clf.load_the_save()
        other_clf.load_data()
        all_data = pd.concat([other_clf.data[part] for part in ['train', 'dev', 'test']])
        x, y = self._only_scale(all_data)
        pred_dict = self._make_predictions(x)
        preds = [dict(sorted(pred.items(), key=lambda item: -item[1])) for pred in pred_dict]

        if dict_only:
            return preds

        data = all_data[['filename', 'snippet_x', 'snippet_y', 'rel']]
        for i in range(3):
            top_i = [list(pred.items())[i] for pred in preds]
            data.loc[:, [f'top_{i + 1}']] = [label for label, prob in top_i]
            data.loc[:, [f'top_{i + 1}_prob']] = [prob for label, prob in top_i]

        return data

    def load_the_save(self):
        self.__dict__.update(pickle.load(open(self.save_path, 'rb')).__dict__)


if __name__ == '__main__':
    fire.Fire(FeatureRichClassifier)
