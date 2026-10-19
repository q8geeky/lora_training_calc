import unittest
from training_core import TrainingPlan, estimate, starting_repeats


def plan(**changes):
    values = dict(images=23, repeats=10, epochs=7, batch_size=4,
                  gradient_accumulation=2, learning_rate=0.0001,
                  rank=16, alpha=16)
    values.update(changes)
    return TrainingPlan(**values)


class TrainingMathTests(unittest.TestCase):
    def test_partial_batches_and_accumulation(self):
        result = estimate(plan())
        self.assertEqual(result.images_per_epoch, 230)
        self.assertEqual(result.batches_per_epoch, 58)
        self.assertEqual(result.optimizer_steps_per_epoch, 29)
        self.assertEqual(result.total_optimizer_steps, 203)
        self.assertEqual(result.total_image_exposures, 1610)
        self.assertEqual(result.effective_batch_size, 8)

    def test_epochs_reset_accumulation(self):
        result = estimate(plan(images=1, repeats=1, batch_size=8,
                               gradient_accumulation=8, epochs=3))
        self.assertEqual(result.total_optimizer_steps, 3)

    def test_vram_not_equal_batch_size(self):
        self.assertEqual(estimate(plan(batch_size=1)).effective_batch_size, 2)

    def test_bad_inputs_rejected(self):
        for bad in [dict(images=0), dict(batch_size=0),
                    dict(gradient_accumulation=0), dict(learning_rate=float("nan")),
                    dict(epochs=-1)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimate(plan(**bad))

    def test_heuristic_bounded(self):
        self.assertEqual(starting_repeats(20), 10)
        self.assertEqual(starting_repeats(500), 1)
        self.assertEqual(starting_repeats(1), 40)


if __name__ == "__main__":
    unittest.main()
