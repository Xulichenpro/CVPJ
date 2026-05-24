import unittest

import numpy as np

from utils.metrics import eval_func


class MetricsCCVIDTest(unittest.TestCase):
    def test_ccvid_eval_keeps_same_camera_positive_matches(self):
        distmat = np.array([[0.1, 0.9]], dtype=np.float32)
        q_pids = np.array([1])
        g_pids = np.array([1, 2])
        q_camids = np.array([0])
        g_camids = np.array([0, 0])

        cmc, mAP = eval_func(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            dataset_name="ccvid",
        )

        self.assertEqual(cmc[0], 1.0)
        self.assertEqual(mAP, 1.0)


if __name__ == "__main__":
    unittest.main()
