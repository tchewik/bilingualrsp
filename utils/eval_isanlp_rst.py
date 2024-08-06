import os
import pickle
from pathlib import Path
from tqdm import tqdm
import fire

from isanlp import PipelineCommon
from isanlp.processor_remote import ProcessorRemote
from isanlp.processor_razdel import ProcessorRazdel

from dmrst_parser.data_manager import DataManager


class IsaNLPRSTEvaluator:
    def __init__(self, syntax_ip, syntax_port, rst_ip, rst_port):

        address_syntax = (syntax_ip, syntax_port)
        address_rst = (rst_ip, rst_port)

        self.ppl_ru = PipelineCommon([
            (ProcessorRazdel(), ['text'],
             {'tokens': 'tokens',
              'sentences': 'sentences'}),
            (ProcessorRemote(address_syntax[0], address_syntax[1], '0'),
             ['tokens', 'sentences'],
             {'lemma': 'lemma',
              'morph': 'morph',
              'syntax_dep_tree': 'syntax_dep_tree',
              'postag': 'postag'}),
            (ProcessorRemote(address_rst[0], address_rst[1], 'default'),
             ['text', 'tokens', 'sentences', 'postag', 'morph', 'lemma', 'syntax_dep_tree'],
             {'rst': 'rst'})
        ])

    def evaluate(self, corpus, lang, reparse=False):
        corpus_dump = os.path.join('data', f'data_manager_{corpus.lower()}.pickle')
        dm = DataManager(corpus=corpus).from_pickle(corpus_dump)
        _, _, test = dm.get_data(lang=lang)

        annotation_path = Path(os.path.join('data', 'isanlp_rst_predictions', corpus, lang))

        if reparse or not os.path.isdir(annotation_path):
            self.parse(annotation_path, dm, test)

    def parse(self, annotation_path, dm, test):
        annotation_path.mkdir(parents=True, exist_ok=True)
        for filename, tokens in tqdm(zip(dm.corpus['test'], test.input_sentences), total=len(test.input_sentences)):
            text = ' '.join(tokens).replace('<P>', '\n').strip()
            nlp_annot = self.ppl_ru(text)
            with open(annotation_path.joinpath(filename + '.pkl'), 'wb') as f:
                pickle.dump(nlp_annot, f)


if __name__ == '__main__':
    fire.Fire(IsaNLPRSTEvaluator)
