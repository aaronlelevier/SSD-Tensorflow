import unittest

import numpy as np
import tensorflow as tf

from tf_extended import metrics


class MetricsTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [1, True],
            [2, True],
            [3, False],
            [4, False],
            [5, False],
            [6, True],
            [7, True],
            [8, False],
            [9, False],
            [10, True]
        ])

    def _precision(self):
        raw_precision_arr = [
            1.0,
            1.0,
            0.6666666666666666,
            0.5,
            0.4,
            0.5,
            0.5714285714285714,
            0.5,
            0.4444444444444444,
            0.5]

        arr = []
        count = 0
        for idx, x in enumerate(self.X):
            if x[1] == 1:
                count += 1.
            arr.append(count / (idx+1))
        precision_arr = np.array(arr)

        assert np.allclose(precision_arr, raw_precision_arr)

        return tf.constant(precision_arr, dtype=tf.float64)

    def _recall(self):
        raw_recall_arr = [0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 1.]

        pos_labels = 5
        arr = []
        count = 0
        for x in self.X:
            if x[1] == 1:
                count += 1.
            arr.append(count / pos_labels)
        recall_arr = np.array(arr)

        assert np.allclose(recall_arr, raw_recall_arr)

        return tf.constant(recall_arr, dtype=tf.float64)

    def test_average_precision_voc07(self):
        precision = tf.constant(recall_arr, dtype=tf.float64)
        recall = self._recall()
        sess = tf.Session()

        ret = sess.run(metrics.average_precision_voc07(precision, recall))

        assert np.allclose([ret], [0.7532467532467532])
