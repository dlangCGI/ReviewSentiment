import unittest
import json
from review_classification.predict import make_prediction, TEXTCOLUMN


class Test(unittest.TestCase):
    """
    Basic unittest class to instantiate testcases.
    Here used for more challenging tests for the classifier.
    """
    def test_0_advanced_positive(self):
        """
        Test advanced positive comments classification
        """
        testinput = {TEXTCOLUMN: ['The service was not bad. Insane phone!',
                                  'Although terrible costumer servie, i '
                                  'received a great product!']}
        y_pred = make_prediction(json.dumps(testinput))['predictions']
        self.assertEqual(y_pred.tolist(), [1, 1])

    def test_1_advanced_negative(self):
        """
        Test advanced negative comments classification
        """
        testinput = {TEXTCOLUMN: ['Was not happy... Next time buying a '
                                  'product with more good reviews.',
                                  'The price was good, but the quality '
                                  'really is not. No recommendation!']}
        y_pred = make_prediction(json.dumps(testinput))['predictions']
        self.assertListEqual(y_pred.tolist(), [0, 0])

    def test_2_not_clean_input(self):
        """
        Test classification for comments that have to be cleaned
        """
        testinput = {TEXTCOLUMN: ['Don\'t know whatt to say.. very '
                                  'dissapppointed.', 'Alrigt     but not '
                                                     'speciffically good. '
                                                     'Overall only 2 stars so '
                                                     'no recommendatin.',
                                  'woooow, greeat!']}
        y_pred = make_prediction(json.dumps(testinput))['predictions']
        self.assertEqual(y_pred.tolist(), [0, 0, 1])


if __name__ == '__main__':
    # start tests
    unittest.main()
