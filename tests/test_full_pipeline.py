"""
Regression test for the DSG pipeline.
Verifies that DSGDetector works with a small model.
"""

import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar.icml.dsg_detector import DSGDetector
from ag_sar.config import DSGConfig


class TestDSGPipeline(unittest.TestCase):
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

    def test_basic_detection(self):
        if self.model is None:
            self.skipTest("Model not available")

        config = DSGConfig()
        detector = DSGDetector(self.model, self.tokenizer, config)

        result = detector.detect(
            question="What is 2+2?",
            context="Basic arithmetic: 2+2=4.",
            max_new_tokens=10
        )

        self.assertIsNotNone(result.generated_text)
        self.assertTrue(len(result.token_risks) > 0)
        self.assertTrue(0 <= result.response_risk <= 1.0)
        print(f"DSG: Text='{result.generated_text.strip()}', Risk={result.response_risk}")

if __name__ == "__main__":
    unittest.main()
