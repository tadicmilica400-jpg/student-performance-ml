"""Synthetic checks, not evidence of real-data model quality."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from main import build_pipeline, evaluate


class PipelineTests(unittest.TestCase):
    def test_training_statistics_and_unknown_category(self):
        X = pd.DataFrame({'hours': [1., 2., 3., 4.], 'group': ['a', 'b', 'a', 'b']})
        pipeline = build_pipeline().fit(X, [2., 4., 6., 8.])
        scale = pipeline.named_steps['preprocessing'].named_transformers_['numeric'].named_steps['scale']
        self.assertAlmostEqual(scale.mean_[0], 2.5)
        pred = pipeline.predict(pd.DataFrame({'hours': [1000.], 'group': ['unseen']}))
        self.assertTrue(np.isfinite(pred).all())
        self.assertAlmostEqual(scale.mean_[0], 2.5)

    def test_end_to_end_with_missing_features(self):
        rng = np.random.default_rng(42)
        hours = rng.normal(size=100)
        df = pd.DataFrame({'hours': hours, 'group': ['a', 'b'] * 50,
                           'Exam_Score': 70 + 4 * hours + rng.normal(size=100)})
        df.loc[0, 'hours'] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.csv'
            df.to_csv(path, index=False)
            result = evaluate(path)
        self.assertEqual(result['train_rows'], 80)
        self.assertEqual(result['test_rows'], 20)
        self.assertTrue(np.isfinite(result['holdout_mae']))
        self.assertEqual(len(result['cv_candidates']), 21)


if __name__ == '__main__':
    unittest.main()
