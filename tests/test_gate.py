import unittest

import numpy as np

from lad.experiment import audit_groups
from lad.scoring import Components, compare_objects, normal_bank, scores


def scene(probabilities):
    boxes = [[i / 10, 0.1, i / 10 + 0.07, 0.25] for i in range(len(probabilities))]
    return Components.load([{"p": p, "box": b} for p, b in zip(probabilities, boxes)], 3)


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.probs = [[.998, .001, .001]] * 2 + [[.001, .998, .001]] * 2
        self.normal = scene(self.probs)
        self.bank = normal_bank([({"split": "train", "label": 0}, self.normal)])

    def test_clean_and_uncertain_identity(self):
        self.assertAlmostEqual(scores(self.normal, self.bank)["joint"], 0)
        uncertain = scene([*self.probs[:3], [.51, .48, .01]])
        result = scores(uncertain, self.bank)
        self.assertAlmostEqual(result["joint"], np.log(.51/.48))
        self.assertGreater(result["hard"], result["joint"])
        self.assertAlmostEqual(result["normal_cost"] - result["free"], result["joint"])

    def test_weak_existence_and_real_missing(self):
        weak = scene([*self.probs[:3], [.01, .48, .51]])
        missing = scene(self.probs[:3])
        self.assertAlmostEqual(scores(weak, self.bank)["joint"], np.log(.51/.48))
        self.assertEqual(scores(missing, self.bank)["joint"], 3)
        self.assertEqual(scores(scene([]), self.bank)["joint"], 12)

    def test_false_positive_and_confident_duplicate(self):
        low = scene([*self.probs, [.46, .01, .53]])
        duplicate = scene([*self.probs, self.probs[0]])
        self.assertEqual(scores(low, self.bank)["joint"], 0)
        self.assertEqual(scores(duplicate, self.bank)["joint"], 3)
        self.assertEqual(scores(duplicate, self.bank)["forced"], 0)

    def test_gt_comparison_and_split_guards(self):
        stats = compare_objects(scene(self.probs[:3]), self.normal)
        self.assertEqual((stats["missed"], stats["wrong_type"]), (1, 0))
        with self.assertRaises(ValueError):
            normal_bank([({"label": 0, "split": "test"}, self.normal)])
        with self.assertRaises(ValueError):
            audit_groups([{"group": "a"}], [{"group": "a"}], True)
        with self.assertRaises(ValueError):
            audit_groups([{"group": None}], [{"group": "b"}], True)


if __name__ == "__main__":
    unittest.main()
