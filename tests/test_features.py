import unittest

import numpy as np
import pandas as pd

from virtual_cell.features import FeatureEncoder


class FeatureEncoderTest(unittest.TestCase):
    def test_unseen_entities_map_to_unknown(self) -> None:
        train = pd.DataFrame(
            {
                "Strains": ["A", "B"],
                "perturbation_no_concentration": ["Water", "Drug"],
                "Medium": ["M", "M"],
                "Temperature": [30, 30],
                "data_source": ["S", "S"],
                "instrument": ["I", "I"],
                "Yeast_cell_plate": ["P", "P"],
                "pert_time": [15, 30],
            }
        )
        encoder = FeatureEncoder.fit(train, ["Water", "DMSO"], "Quality Control")
        unseen = train.iloc[[0]].copy()
        unseen["Strains"] = "UNSEEN"
        unseen["perturbation_no_concentration"] = "NEW_DRUG"
        transformed = encoder.transform(unseen)
        self.assertEqual(int(transformed["Strains"][0]), 0)
        self.assertEqual(int(transformed["perturbation_no_concentration"][0]), 0)
        self.assertEqual(float(transformed["is_treatment"][0, 0]), 1.0)
        self.assertTrue(np.isfinite(transformed["time_features"]).all())


if __name__ == "__main__":
    unittest.main()

