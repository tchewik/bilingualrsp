""" This script allows for collecting the statistics for Table 1. """

import os
import shutil
from pathlib import Path

import fire
import pandas as pd
import razdel
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from isanlp.annotation_rst import DiscourseUnit

from dmrst_parser.data_manager import DataManager
from dmrst_parser.src.parser.data import Data


def select_genre(data, filenames, genre):
    """ Given the Data object, filter the documents corresponding to given genre. """

    _filenames = []
    _tokens = []
    _edu_breaks = []
    _golden_metrics = []

    for i, filename in enumerate(filenames):
        if genre in filename:
            _filenames.append(filename)
            _tokens.append(data.input_sentences[i])
            _edu_breaks.append(data.edu_breaks[i])
            _golden_metrics.append(data.golden_metric[i])

    _data = Data(input_sentences=_tokens, edu_breaks=_edu_breaks, golden_metric=_golden_metrics,
                decoder_input=None, relation_label=None, parsing_breaks=None)

    return _data, _filenames


@Language.component('set_custom_boundaries')
def set_custom_boundaries(doc):
    new_paragraph = '<P>'
    not_sentence_ending = ['(']
    for token in doc[:-1]:
        if token.text == new_paragraph:
            # It should be placed at the end of the sentence
            doc[token.i].is_sent_start = False
            if token.i < len(doc) - 1:
                doc[token.i + 1].is_sent_start = True

        if token.text in not_sentence_ending:
            if token.i < len(doc) - 1:
                if doc[token.i + 1].is_sent_start:
                    doc[token.i + 1].is_sent_start = False
                    doc[token.i].is_sent_start = True

        # Blah blah . [ ref ]   for some reason by default in spacy splits as SENTENCE . [ <split> number ] ...
        if token.text in ['.', '"'] and token.i < len(doc) - 4:
            if doc[token.i + 1].text == '[':
                if doc[token.i + 2].is_sent_start and doc[token.i + 2].text.isnumeric():
                    if doc[token.i + 3].text == ']':
                        # Move sentence boundary to the next ]
                        doc[token.i + 2].is_sent_start = False
                        doc[token.i + 4].is_sent_start = True

        # For heaven ’s sake , call me Betty . ” “
        if token.text == '“' and token.i < len(doc) - 2:
            if doc[token.i + 1].is_sent_start:
                # Move sentence start to the “
                doc[token.i + 1].is_sent_start = False
                doc[token.i].is_sent_start = True

    return doc


class CustomTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, words):
        words = words.split(' ')
        return Doc(self.vocab, words=words)


class CorpStats:
    def __init__(self, corpus: str, lang: str):
        corpus_dump = os.path.join('data', f'data_manager_{corpus.lower()}.pickle')
        if not os.path.isfile(corpus_dump):
            shutil.rmtree(os.path.join('data', f'{corpus.lower().replace("-", "")}_prepared'))

            cv = corpus == 'RST-DT'
            dp = DataManager(corpus=corpus, cross_validation=cv)
            dp.from_rs3()
            dp.save(corpus_dump)

        self.dm = DataManager(corpus=corpus).from_pickle(corpus_dump)
        self.lang = lang

        self.pairs_dump_path = Path(os.path.join('data', 'relation_pairs'))
        self.pairs_dump_path.mkdir(parents=True, exist_ok=True)

        if lang == 'en':
            spacy_model = 'en_core_web_sm'
            self._spacy = spacy.load(spacy_model, disable=['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer'])
            self._spacy.disable_pipe("parser")
            self._spacy.tokenizer = CustomTokenizer(self._spacy.vocab)

            # case = [{ORTH: '<P>'}]
            # self._spacy.tokenizer.add_special_case('<P>', case)
            self._spacy.add_pipe("sentencizer", before='tok2vec')
            self._spacy.add_pipe('set_custom_boundaries', after="sentencizer")

        self._du_id = 0

    @staticmethod
    def edus_in_isanlp_format(tokens: list, edu_breaks: list):
        """ Collects isanlp.DiscourseUnit EDUs from the list of document tokens and edu_breaks. """
        prev_break = 0
        prev_chr_end = 0
        edus = []
        for i, brk in enumerate(edu_breaks):
            edu = DiscourseUnit(
                id=i,
                text=' '.join(tokens[prev_break:brk + 1]).strip(),
                start=prev_chr_end
            )
            edu.end = edu.start + len(edu.text)
            prev_chr_end = edu.end + 1
            prev_break = brk + 1
            edus.append(edu)
        return edus

    @staticmethod
    def tree_string_to_list(description):
        """ Parses the description in a convenient list format """
        rels = []
        for rel in description.split(' '):
            left, right = rel.split(',')
            left_start, left_label, left_end = left[1:].split(':')
            right_start, right_label, right_end = right[:-1].split(':')
            nuclearity = left_label[0] + right_label[0]
            relation = left_label.split('=')[1] if nuclearity == 'SN' else right_label.split('=')[1]
            rels.append((int(left_start) - 1,  # left DU's start EDU
                         int(left_end) - 1,  # left DU's end EDU
                         relation,  # rhetorical relation
                         nuclearity,  # nuclearity of the relation
                         int(right_start) - 1,  # right DU's start EDU
                         int(right_end) - 1))  # right DU's end EDU
        return rels

    @staticmethod
    def get_child(start, end, rels):
        """ Given the non-terminal DU span find the position of its string description """
        for idx, rel in enumerate(rels):
            if rel[0] == start and rel[-1] == end:
                return idx

    @staticmethod
    def collect_relations(edus, rels, filename=''):
        result = []
        for rel in rels:
            left_start, left_end, relation, nuclearity, right_start, right_end = rel
            left_txt = ' '.join([edu.text for edu in edus[left_start:left_end + 1]])
            right_txt = ' '.join([edu.text for edu in edus[right_start:right_end + 1]])
            result.append(
                [left_txt, right_txt, edus[left_start].start, edus[right_start].start, relation, nuclearity, filename])

        return result

    def _corpus_part_info(self, data, filenames, genre=None):

        # (Optionally) select the documents of specific genre
        if genre:
            data, filenames = select_genre(data, filenames, genre)

        data_edus = [self.edus_in_isanlp_format(data.input_sentences[i], data.edu_breaks[i])
                     for i in range(len(data.input_sentences))]

        data_rels = [self.tree_string_to_list(data.golden_metric[i]) for i in range(len(data.golden_metric))]
        data_pairs = []
        for edus, rels, filename in zip(data_edus, data_rels, filenames):
            data_pairs += self.collect_relations(edus, rels, filename)

        data_pairs = pd.DataFrame(data_pairs,
                                  columns=['snippet_x', 'snippet_y', 'loc_x', 'loc_y',
                                           'category_id', 'order', 'filename'])

        return {
            'texts': data.input_sentences,
            'num_trees': len(data.input_sentences),
            'edus': data_edus,
            'rels': data_rels,
            'pairs': data_pairs,
            'tree_lens': [len(tree) for tree in data.input_sentences]
            }

    @staticmethod
    def edus_stats(edus: list, num_trees: int):
        all_edus = 0
        for edus in edus:
            all_edus += len(edus)

        print('Number of EDUs:', all_edus)
        print('EDUs per tree:', all_edus / num_trees)

    @staticmethod
    def tokens_stats(tree_lens: list):
        lens = pd.Series(tree_lens)
        print(f'Tokens per tree: min = {lens.min()}, max = {lens.max()}, median = {lens.median()}')

    @staticmethod
    def rels_stats(pairs: pd.DataFrame):
        num_classes = (pairs.category_id + pairs.order).unique().shape[0]
        print('Number of classes:', num_classes)
        print('Number of DU pairs:', pairs.shape[0])

    def collect_sent_spans(self, tokens):
        sent_spans = []
        text = ' '.join(tokens)

        # (word_start_char, word_end_char+1) for each token
        word_offsets = []
        cur_char = 0
        for word in text.split():
            word_offsets.append((cur_char, cur_char + len(word)))
            cur_char += len(word) + 1

        if self.lang == 'en':
            sentences = self._spacy(text).sents
        elif self.lang == 'ru':
            sentences = razdel.sentenize(text)

        for sent in sentences:
            # Collect the sent_offsets for matching with word_offsets
            if self.lang == 'en':
                sent_offset = (sent.start_char, sent.end_char)
            elif self.lang == 'ru':
                sent_offset = (sent.start, sent.stop)

            for i, word in enumerate(word_offsets):
                if word[0] >= sent_offset[0]:
                    start_token_offset = i
                    for j, word in enumerate(word_offsets):
                        if word[1] >= sent_offset[1]:
                            end_token_offset = j + 1
                            break
                    break

            sent_spans.append((start_token_offset, end_token_offset))

        return sent_spans

    @staticmethod
    def edu_spans_to_token_spans(rels, edus):
        # Spans in EDUs are given for chars; find the token spans of EDUs
        edus_token_spans = []
        edu_start = 0
        for edu in edus:
            edu_len = len(edu.text.strip().split())
            edus_token_spans.append((edu_start, edu_start + edu_len))
            edu_start += edu_len

        # Recount for non-elementary discourse units
        token_rels = []
        for rel in rels:
            start = rel[0]
            end = rel[-1]
            token_rels.append((edus_token_spans[start][0], edus_token_spans[end][1]))

        return token_rels, edus_token_spans

    def sentence_stats(self, tokens, edus, rels):
        sent_spans = [self.collect_sent_spans(tokens) for tokens in tokens]

        all_token_spans = [self.edu_spans_to_token_spans(rels_list, edus_list) for rels_list, edus_list in
                           zip(rels, edus)]

        # For all discourse units
        all_token_spans_wedus = [spans[0] + spans[1] for spans in all_token_spans]
        spanned_sentences = [self.count_spanned_sentences(ats, ss)
                             for ats, ss in zip(all_token_spans_wedus, sent_spans)]

        ratio = sum([span[0] for span in spanned_sentences]) / sum([span[1] for span in spanned_sentences])
        print('Spanned sentences ratio:', ratio)

        # Without EDUs
        all_token_spans_woedus = [spans[0] for spans in all_token_spans]
        all_edus_spans = [spans[1] for spans in all_token_spans]
        sent_spans_woedus = []
        for doc, edus in zip(sent_spans, all_edus_spans):
            sent_spans_woedus.append([sent_span for sent_span in doc if sent_span not in edus])
        spanned_sentences = [self.count_spanned_sentences(ats, ss)
                             for ats, ss in zip(all_token_spans_woedus, sent_spans_woedus)]
        ratio = sum([span[0] for span in spanned_sentences]) / sum([span[1] for span in spanned_sentences])
        print('Spanned sentences ratio w/o EDUs:', ratio)

    @staticmethod
    def count_spanned_sentences(all_token_spans, sent_spans):
        spanned_sentences = sum([span in all_token_spans for span in sent_spans])
        return spanned_sentences, len(sent_spans)

    def collect(self, save_pairs=False, genre=None):
        if self.dm.corpus_name == 'RST-DT':
            train, dev, test = self.dm.get_fold(0, lang='en')
            train_info = self._corpus_part_info(train, self.dm.folds[0]['train'], genre)
            dev_info = self._corpus_part_info(dev, self.dm.folds[0]['dev'], genre)
            test_info = self._corpus_part_info(test, self.dm.folds[0]['test'], genre)
        else:
            train, dev, test = self.dm.get_data(lang=self.lang)
            train_info = self._corpus_part_info(train, self.dm.corpus['train'], genre)
            dev_info = self._corpus_part_info(dev, self.dm.corpus['dev'], genre)
            test_info = self._corpus_part_info(test, self.dm.corpus['test'], genre)

        full_texts = train_info.get('texts') + dev_info.get('texts') + test_info.get('texts')
        full_edus = train_info.get('edus') + dev_info.get('edus') + test_info.get('edus')
        full_num_trees = train_info.get('num_trees') + dev_info.get('num_trees') + test_info.get('num_trees')
        full_tree_lens = train_info.get('tree_lens') + dev_info.get('tree_lens') + test_info.get('tree_lens')
        full_rels = train_info.get('rels') + dev_info.get('rels') + test_info.get('rels')
        full_pairs = pd.concat([train_info.get('pairs'), dev_info.get('pairs'), test_info.get('pairs')])

        self.edus_stats(full_edus, full_num_trees)
        self.tokens_stats(full_tree_lens)
        self.rels_stats(full_pairs)
        self.sentence_stats(full_texts, full_edus, full_rels)

        if save_pairs:
            train_info.get('pairs').to_feather(
                self.pairs_dump_path.joinpath(f'{self.dm.corpus_name}_{self.lang}_train.fth'))
            dev_info.get('pairs').to_feather(
                self.pairs_dump_path.joinpath(f'{self.dm.corpus_name}_{self.lang}_dev.fth'))
            test_info.get('pairs').to_feather(
                self.pairs_dump_path.joinpath(f'{self.dm.corpus_name}_{self.lang}_test.fth'))


if __name__ == '__main__':
    fire.Fire(CorpStats)
