"""离线检查契约；仅编写，由用户运行。"""
import unittest
import numpy as np
import torch
from common import prediction_rows
from data import build_vocabulary, chunk_ids
from encoders import CharacterEncoder
from model import MultimodalModel
from experiments import selection_scopes


class Contracts(unittest.TestCase):
    def test_vocabulary_is_training_only(self):
        snapshots = [{"news": "训练甲", "factor": "需求乙"}, {"news": "独有龘", "factor": ""}]
        vocab = build_vocabulary(snapshots, [0], 100)
        self.assertNotIn("龘", vocab)
        self.assertIn("甲", vocab)

    def test_full_text_and_cnn_boundary_windows(self):
        ids = list(range(1, 1001))
        chunks = chunk_ids(ids, 64, 4)
        self.assertEqual(chunks[0] + sum([c[4:] for c in chunks[1:]], []), ids)
        windows = {tuple(c[i:i+5]) for c in chunks for i in range(len(c)-4)}
        self.assertTrue(all(tuple(ids[i:i+5]) in windows for i in range(len(ids)-4)))

    def test_textcnn_masks_padding(self):
        encoder = CharacterEncoder(16, 8, "textcnn").eval()
        x = torch.randn(2, 9, 16)
        lengths = torch.tensor([5, 9])
        a = encoder(x, lengths)
        x[0, 5:] = 1000
        b = encoder(x, lengths)
        torch.testing.assert_close(a[0], b[0])

    def test_all_frameworks_orders_and_price_gradient(self):
        snapshots = [
            {"news_ids": [3, 4, 5, 6], "factor_ids": [4, 5], "flags": [1, 1, 0, 0, 0]},
            {"news_ids": [], "factor_ids": [], "flags": [0, 0, 0, 0, 0]},
        ]
        for name in ["ucmib_pns", "graph_gated", "hierarchical", "parallel", "lcmf"]:
            for order in ["late", "early"]:
                recipe = {"id": "check", "order": order, "paired": True,
                          "price": "gru", "news": "textcnn", "factor": "textcnn", "joint": "textcnn"}
                net = MultimodalModel(name, recipe, snapshots, 10, 8, 8, 32, 8)
                price = torch.randn(2, 3, 2, requires_grad=True)
                out = net(price, torch.tensor([[0, 1, 0], [1, 0, 1]]), return_features=True)
                self.assertEqual(tuple(out["prediction"].shape), (2,))
                self.assertIn("fusion_features", out)
                net.loss(out, torch.randn(2), torch.arange(2)).backward()
                self.assertTrue(torch.isfinite(price.grad).all())
                self.assertGreater(float(price.grad.abs().sum()), 0)

    def test_unmatched_filters_do_not_claim_paired_comparison(self):
        recipes = [
            {"id": "late_a", "order": "late", "paired": True, "pair_id": "a"},
            {"id": "early_b", "order": "early", "paired": True, "pair_id": "b"},
        ]
        scopes = selection_scopes(recipes)
        self.assertIn("unpaired_late", scopes["late_a"])
        self.assertIn("unpaired_early", scopes["early_b"])
        self.assertNotIn("paired_late", scopes["late_a"])
        self.assertNotIn("paired_early", scopes["early_b"])

    def test_output_price_change_and_direction(self):
        rows = prediction_rows(["2025-01-02"], np.array([110.]), np.array([100.]), np.array([.5]), 20.)
        self.assertEqual(rows[0]["predicted_price"], 110.)
        self.assertEqual(rows[0]["predicted_change"], 10.)
        self.assertEqual(rows[0]["predicted_change_percent"], 10.)
        self.assertEqual(rows[0]["predicted_direction"], "上涨")


if __name__ == "__main__":
    unittest.main()


