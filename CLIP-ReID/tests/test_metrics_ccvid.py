import unittest

import numpy as np
import torch

from utils.metrics import euclidean_distance, eval_func, eval_func_chunked


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

    def test_chunked_eval_matches_full_distance_eval_for_ccvid(self):
        qf = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9, 0.1],
            ],
            dtype=torch.float32,
        )
        gf = torch.tensor(
            [
                [1.0, 0.1],
                [0.1, 1.0],
                [0.8, 0.2],
                [0.0, 0.9],
            ],
            dtype=torch.float32,
        )
        q_pids = np.array([1, 2, 1])
        g_pids = np.array([1, 2, 3, 1])
        q_camids = np.array([0, 0, 0])
        g_camids = np.array([0, 0, 0, 0])

        full_cmc, full_mAP = eval_func(
            euclidean_distance(qf, gf),
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            dataset_name="ccvid",
        )

        chunked_cmc, chunked_mAP = eval_func_chunked(
            qf,
            gf,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=4,
            dataset_name="ccvid",
            query_chunk_size=2,
        )

        np.testing.assert_allclose(chunked_cmc, full_cmc[:4])
        self.assertAlmostEqual(chunked_mAP, full_mAP)


if __name__ == "__main__":
    unittest.main()
