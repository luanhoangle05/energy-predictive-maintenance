"""Minimal checks that the project package is importable."""


def test_project_package_imports() -> None:
    """Import each scaffold module without executing pipeline work."""
    from src.data import load_data, validate_data
    from src.evaluation import evaluate_model
    from src.features import build_features
    from src.models import predict, train_model

    assert all(
        module is not None
        for module in (
            load_data,
            validate_data,
            evaluate_model,
            build_features,
            predict,
            train_model,
        )
    )
