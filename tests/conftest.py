"""Shared test helpers: the worked example, ready to run with small changes."""

import pytest

from sizing import models
from sizing.engine import Inputs, size_workload

# Not a real model: a made-up big model, only used to test the "spans servers" rules.
FAKE_BIG_MODEL = {"name": "Fake big model (test only)", "params_b": 400,
                  "layers": 120, "kv_heads": 8, "head_size": 128}


def worked_inputs(**changes):
    """Return the worked-example inputs, with any changes applied."""
    values = dict(model_name="Llama 3 70B class", users=200, tokens_per_user=4_096,
                  precision="FP16", target_tokens_per_s=20, gpu_name="H100 SXM",
                  backup_mode="none")
    values.update(changes)
    return Inputs(**values)


@pytest.fixture(scope="session")
def gpus():
    return models.load_gpus()


@pytest.fixture(scope="session")
def server():
    return models.find_by_name(models.load_servers(), "Dell PowerEdge XE9680")


@pytest.fixture(scope="session")
def run(gpus, server):
    """Return a function that sizes the worked example with any changes applied."""
    def _run(model=None, overrides=None, **changes):
        inputs = worked_inputs(**changes)
        gpu = models.find_by_name(gpus, inputs.gpu_name)
        return size_workload(inputs, gpu, server, overrides=overrides, model=model)
    return _run
