"""
Tests for experiment data loaders.

Verifies that dataset loaders implement the EvaluationDataset interface
correctly and can load data from HuggingFace.
"""

import pytest

from experiments.data.base import EvaluationDataset, EvaluationSample
from experiments.data.halueval import HaluEvalDataset
from experiments.data.ragtruth import RAGTruthDataset
from experiments.data.truthfulqa import TruthfulQADataset
from experiments.data.wikitext import WikiTextDataset
from experiments.data.fava import FAVADataset


class TestEvaluationSample:
    """Tests for EvaluationSample dataclass."""

    def test_valid_sample(self):
        """Should create valid sample with label 0 or 1."""
        sample = EvaluationSample(
            prompt="What is 2+2?",
            response="4",
            label=0,
        )
        assert sample.prompt == "What is 2+2?"
        assert sample.response == "4"
        assert sample.label == 0

    def test_label_validation(self):
        """Should reject invalid labels."""
        with pytest.raises(ValueError):
            EvaluationSample(
                prompt="Test",
                response="Response",
                label=2,  # Invalid
            )

    def test_metadata_optional(self):
        """Metadata should be optional."""
        sample = EvaluationSample(
            prompt="Test",
            response="Response",
            label=1,
        )
        assert sample.metadata is None

        sample_with_meta = EvaluationSample(
            prompt="Test",
            response="Response",
            label=1,
            metadata={"source": "test"},
        )
        assert sample_with_meta.metadata == {"source": "test"}


class TestHaluEvalDataset:
    """Tests for HaluEvalDataset loader."""

    def test_instantiation(self):
        """Should instantiate without loading."""
        ds = HaluEvalDataset(variant="qa", num_samples=10, seed=42)
        assert ds.variant == "qa"
        assert ds.num_samples == 10

    def test_invalid_variant(self):
        """Should reject invalid variants."""
        with pytest.raises(ValueError):
            HaluEvalDataset(variant="invalid")

    def test_valid_variants(self):
        """Should accept all valid variants."""
        for variant in ["qa", "summarization", "dialogue"]:
            ds = HaluEvalDataset(variant=variant)
            assert ds.variant == variant

    def test_not_loaded_error(self):
        """Should raise error if iterating before load."""
        ds = HaluEvalDataset(variant="qa")
        with pytest.raises(RuntimeError):
            list(ds)

    @pytest.mark.slow
    def test_load_qa(self):
        """Should load QA variant from HuggingFace."""
        ds = HaluEvalDataset(variant="qa", num_samples=5, seed=42)
        ds.load()

        assert len(ds) <= 5
        assert all(isinstance(s, EvaluationSample) for s in ds)


class TestRAGTruthDataset:
    """Tests for RAGTruthDataset loader."""

    def test_instantiation(self):
        """Should instantiate without loading."""
        ds = RAGTruthDataset(task_type="QA", num_samples=10, seed=42)
        assert ds.task_type == "QA"

    def test_invalid_task_type(self):
        """Should reject invalid task types."""
        with pytest.raises(ValueError):
            RAGTruthDataset(task_type="invalid")

    def test_valid_task_types(self):
        """Should accept all valid task types."""
        for task_type in ["QA", "Summary", "Data2txt", None]:
            ds = RAGTruthDataset(task_type=task_type)
            assert ds.task_type == task_type


class TestTruthfulQADataset:
    """Tests for TruthfulQADataset loader."""

    def test_instantiation(self):
        """Should instantiate without loading."""
        ds = TruthfulQADataset(num_samples=10, seed=42)
        assert ds.num_samples == 10


class TestWikiTextDataset:
    """Tests for WikiTextDataset loader."""

    def test_instantiation(self):
        """Should instantiate without loading."""
        ds = WikiTextDataset(num_samples=10, seed=42)
        assert ds.num_samples == 10

    def test_length_filters(self):
        """Should respect length filters."""
        ds = WikiTextDataset(min_length=100, max_length=200)
        assert ds.min_length == 100
        assert ds.max_length == 200


class TestFAVADataset:
    """Tests for FAVADataset loader."""

    def test_instantiation(self):
        """Should instantiate without loading."""
        ds = FAVADataset(num_samples=10, seed=42)
        assert ds.num_samples == 10


class TestDatasetInterface:
    """Tests for EvaluationDataset interface compliance."""

    @pytest.mark.parametrize("dataset_class,kwargs", [
        (HaluEvalDataset, {"variant": "qa"}),
        (RAGTruthDataset, {}),
        (TruthfulQADataset, {}),
        (WikiTextDataset, {}),
        (FAVADataset, {}),
    ])
    def test_is_evaluation_dataset(self, dataset_class, kwargs):
        """All datasets should be EvaluationDataset instances."""
        ds = dataset_class(**kwargs)
        assert isinstance(ds, EvaluationDataset)

    @pytest.mark.parametrize("dataset_class,kwargs", [
        (HaluEvalDataset, {"variant": "qa"}),
        (RAGTruthDataset, {}),
        (TruthfulQADataset, {}),
        (WikiTextDataset, {}),
        (FAVADataset, {}),
    ])
    def test_has_required_methods(self, dataset_class, kwargs):
        """All datasets should have required methods."""
        ds = dataset_class(**kwargs)

        assert hasattr(ds, "load")
        assert hasattr(ds, "get_statistics")
        assert hasattr(ds, "__iter__")
        assert hasattr(ds, "__len__")

    @pytest.mark.parametrize("dataset_class,kwargs", [
        (HaluEvalDataset, {"variant": "qa"}),
        (RAGTruthDataset, {}),
        (TruthfulQADataset, {}),
        (WikiTextDataset, {}),
        (FAVADataset, {}),
    ])
    def test_get_dataset_name(self, dataset_class, kwargs):
        """All datasets should have _get_dataset_name method."""
        ds = dataset_class(**kwargs)
        name = ds._get_dataset_name()
        assert isinstance(name, str)
        assert len(name) > 0
