import pytest

import face_pipeline

def test_pipeline_initializes():
    """
    Test that the pipeline can load and initialize without error.
    """
    pipeline = face_pipeline.load_pipeline()
    assert pipeline is not None
    assert pipeline._initialized, "Pipeline should be initialized."
