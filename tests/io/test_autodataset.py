from contextlib import nullcontext as does_not_raise

import pytest

from hivegraph.io.autodataset import AutoDataset


@pytest.mark.io
@pytest.mark.parametrize(
    "dataset_name, exception",
    [
        pytest.param(
            "MUTAG",
            does_not_raise(),
            id="MUTAG",
        ),
        pytest.param(
            "MNIST",
            pytest.raises(AssertionError),
            id="MNIST",
        ),
        pytest.param(
            "Peptides-func",
            does_not_raise(),
            id="Peptides-func",
        ),
    ],
)
def test_AutoDataset(dataset_name: str, exception: Exception) -> None:
    with exception:
        dataset = AutoDataset(dataset_name=dataset_name)
        assert dataset.get_dataset() is not None
