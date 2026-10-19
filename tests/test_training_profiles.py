import json
from pathlib import Path
import tempfile
import unittest
from training_core import TrainingPlan
from training_profiles import (
    save_profile, load_profile, parse_profile_document, profile_document
)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.plan = TrainingPlan(images=20, repeats=5, epochs=8, batch_size=2,
                                 gradient_accumulation=3, learning_rate=0.0001,
                                 rank=16, alpha=8)

    def test_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"my_plan.json"
            save_profile(path,self.plan,12.0)
            loaded,vram=load_profile(path)
            self.assertEqual(loaded,self.plan)
            self.assertEqual(vram,12.0)

    def test_failed_save_preserves_existing(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"my_plan.json"
            path.write_text("KEEP ME",encoding="utf-8")
            with self.assertRaises(ValueError):
                save_profile(path,self.plan,float("nan"))
            self.assertEqual(path.read_text(encoding="utf-8"),"KEEP ME")

    def test_reject_unknown_fields_and_version(self):
        item=profile_document(self.plan,8)
        item["version"]=2
        with self.assertRaises(ValueError):
            parse_profile_document(item)
        item=profile_document(self.plan,8)
        item["settings"]["unrecognized_setting"]=1
        with self.assertRaises(ValueError):
            parse_profile_document(item)

    def test_reject_bool_and_invalid_learning_rate(self):
        item=profile_document(self.plan,8)
        item["settings"]["batch_size"]=True
        with self.assertRaises(ValueError):
            parse_profile_document(item)
        item=profile_document(self.plan,8)
        item["settings"]["learning_rate"]="nan"
        with self.assertRaises(ValueError):
            parse_profile_document(item)

    def test_import_not_modify_unknown_file(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"other.json"
            path.write_text('{"random": "data"}')
            with self.assertRaises(ValueError):
                load_profile(path)


if __name__ == "__main__":
    unittest.main()
