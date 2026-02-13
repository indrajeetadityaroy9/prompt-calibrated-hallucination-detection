"""
Regression test for the full AGSAR pipeline.
Verifies that generation works with detection modes.
"""

import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar.engine import AGSAR
from ag_sar.config import DetectorConfig


class TestAGSARPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a tiny model for speed
        cls.model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
            cls.model = AutoModelForCausalLM.from_pretrained(
                cls.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
        except Exception:
            print("SmolLM not found, skipping integration test requiring model download.")
            cls.model = None

    def test_eigenscore_mode(self):
        if self.model is None:
            self.skipTest("Model not available")

        config = DetectorConfig(
            eigenscore_enabled=True,
            semantic_entropy_enabled=False
        )
        detector = AGSAR(self.model, self.tokenizer, config)

        result = detector.generate(
            prompt="What is 2+2?",
            max_new_tokens=10
        )

        self.assertIsNotNone(result.generated_text)
        self.assertTrue(len(result.token_risks) > 0)
        self.assertIsNotNone(result.token_signals[0].eigenscore)
        print(f"EigenScore Mode: Text='{result.generated_text.strip()}', Risk={result.response_risk}")

    def test_basic_generation(self):
        if self.model is None:
            self.skipTest("Model not available")

        config = DetectorConfig(
            eigenscore_enabled=False,
            semantic_entropy_enabled=False,
        )
        detector = AGSAR(self.model, self.tokenizer, config)

        result = detector.generate(
            prompt="What is 2+2?",
            max_new_tokens=10
        )

        self.assertIsNotNone(result.generated_text)
        self.assertTrue(0 <= result.response_risk <= 1.0)
        print(f"Basic Mode: Text='{result.generated_text.strip()}', Risk={result.response_risk}")

if __name__ == "__main__":
    unittest.main()
