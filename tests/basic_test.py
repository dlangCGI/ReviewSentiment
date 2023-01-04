import unittest
import json
from review_classification.predict import make_prediction, TEXTCOLUMN


class Test(unittest.TestCase):
    """
    Basic unittest class to instantiate testcases.
    Here used for easy tests for the classifier.
    """
    def test_0_basic_positive(self):
        """
        Test basic positive comments classification
        """
        testinput = {TEXTCOLUMN: ['This product was great!']}
        y_pred = make_prediction(json.dumps(testinput))['predictions'][0]
        self.assertEqual(y_pred, 1)

    def test_1_multiple_basic_positive(self):
        """
        Test multiple basic positive comments classification
        """
        testinput = {TEXTCOLUMN: ['Good', 'I liked the phone.',
                                   'My son was happy. The price was good.']}
        y_pred = make_prediction(json.dumps(testinput))['predictions']
        self.assertListEqual(y_pred.tolist(), [1, 1, 1])

    def test_2_basic_negative(self):
        """
        Test basic negative comments classification
        """
        testinput = {TEXTCOLUMN: ['This product was terrible!']}
        y_pred = make_prediction(json.dumps(testinput))['predictions'][0]
        self.assertEqual(y_pred, 0)

    def test_3_multiple_basic_mixed(self):
        """
        Test multiple basic mixed comments classification
        """
        testinput = {TEXTCOLUMN: ['I appreciated the service. Will buy again.',
                                  'Never buying a phone again from this shop!',
                                  'I would suggest this nice company '
                                  'to other people.']}
        y_pred = make_prediction(json.dumps(testinput))['predictions']
        self.assertListEqual(y_pred.tolist(), [1, 0, 1])


if __name__ == '__main__':
    # start tests
    unittest.main()
