import pickle

from isanlp.annotation_rst import DiscourseUnit


class DUConverter:
    def __init__(self, predictions_path: str):
        self.predictions_path = predictions_path
        self.du_id = 0
        self.data = self.collect()

    def collect(self):
        """
        Takes the model outputs and converts them into isanlp binary trees.

        Returns:
            List of the predictions as isanlp.DiscourseUnit objects.
        """

        with open(self.predictions_path, 'rb') as f:
            predictions = pickle.load(f)

        data = []
        for i in range(len(predictions['tokens'])):
            edus = self._lists_to_isanlp_format(tokens=predictions['tokens'][i],
                                                edu_breaks=predictions['edu_breaks'][i])
            self.du_id = len(edus)
            rels = self._tree_string_to_list(predictions['spans'][i][0])
            tree = self.construct_tree(0, edus, rels)
            data.append(tree)

        return data

    @staticmethod
    def _lists_to_isanlp_format(tokens, edu_breaks):
        """
        Produces EDUs in isanlp format from the model predictions.

        Args:
            tokens: List of tokens for a document.
            edu_breaks: List of tokens positions with predicted EDU breaks.

        Returns:
            List of the EDUs in isanlp format.
        """

        prev_break = 0
        prev_chr_end = 0
        edus = []
        for i, brk in enumerate(edu_breaks):
            edu = DiscourseUnit(
                id=i,
                text=''.join(tokens[prev_break:brk + 1]).replace('▁', ' ').strip(),
                start=prev_chr_end,
                relation='elementary'
            )
            edu.end = edu.start + len(edu.text)
            prev_chr_end = edu.end + 1
            prev_break = brk + 1
            edus.append(edu)
        return edus

    @staticmethod
    def _tree_string_to_list(description):
        """
        Parses the tree predictions given in a string format.

        Args:
            description: Tree description as a string.

        Returns:
            List of tuples describing constituents.
        """
        rels = []
        for rel in description.split(' '):
            left, right = rel.split(',')
            left_start, left_label, left_end = left[1:].split(':')
            right_start, right_label, right_end = right[:-1].split(':')
            nuclearity = left_label[0] + right_label[0]
            relation = left_label.split('=')[1] if nuclearity == 'SN' else right_label.split('=')[1]
            rels.append((int(left_start) - 1,
                         int(left_end) - 1,
                         relation,
                         nuclearity,
                         int(right_start) - 1,
                         int(right_end) - 1))
        return rels

    @staticmethod
    def _get_child(start, end, rels):
        """
        Selects the discourse unit description for given constituent.

        Args:
            start: DU start position.
            end: DU end position.
            rels: List of tuples describing all the RST tree constituents.

        Returns:
            Index of the given DU in the rels list.
        """

        for idx, rel in enumerate(rels):
            if rel[0] == start and rel[-1] == end:
                return idx

    def construct_tree(self, root, edus, rels):
        """
        Constructs the DiscourseUnit binary tree.

        Args:
            root: Index of the root relation in the rels list.
            edus: List of EDUs as DiscourseUnit objects.
            rels: List of tuples describing all the RST tree constituents.

        Returns:
            Binary DiscourseUnit RST tree.
        """

        left_start, left_end, relation, nuclearity, right_start, right_end = rels[root]

        if left_start == left_end:
            left = edus[left_start]
        else:
            left_root = self._get_child(left_start, left_end, rels)
            left = self.construct_tree(left_root, edus, rels)

        if right_start == right_end:
            right = edus[right_start]
        else:
            right_root = self._get_child(right_start, right_end, rels)
            right = self.construct_tree(right_root, edus, rels)

        self.du_id += 1
        return DiscourseUnit(id=self.du_id,
                             left=left,
                             right=right,
                             relation=relation,
                             nuclearity=nuclearity,
                             start=left.start,
                             end=right.end,
                             text=left.text + ' ' + right.text)
