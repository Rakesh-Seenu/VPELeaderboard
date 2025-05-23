import os
import shutil
import pickle
import pytest
import torch
import tempfile
import torch_geometric
from omegaconf import OmegaConf
from vpeleaderboard.data.src.kg.biobridge_datamodule_hetero import BioBridgeDataModule

# Constants for test environment
PRIMEKG_LOCAL_DIR = "tests/tmp/primekg_test/"
BIOBRIDGE_LOCAL_DIR = "tests/tmp/biobridge_primekg_test/"
CACHE_PATH = "tests/tmp/test_cache.pkl"


import tempfile

@pytest.fixture(scope="module")
def biobridge_config():
    """
    Create a fake Hydra config for BioBridgeDataModule.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        return OmegaConf.create({
            "data": {
                "primekg_dir": os.path.join(tmpdir, "primekg_test"),
                "biobridge_dir": os.path.join(tmpdir, "biobridge_primekg_test"),
                "batch_size": 1,
                "cache_path": os.path.join(tmpdir, "test_cache.pkl")
            },
            "random_link_split": {
                "num_val": 0.1,
                "num_test": 0.1,
                "is_undirected": True,
                "add_negative_train_samples": False,
                "neg_sampling_ratio": 1.0,
                "split_labels": False
            }
        })


@pytest.fixture(scope="module")
def datamodule(biobridge_config):
    """
    Fixture for instantiating and preparing the BioBridgeDataModule.
    """
    # Ensure no previous cache
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    module = BioBridgeDataModule(biobridge_config)
    module.prepare_data()
    return module

def test_prepare_data_creates_cache(datamodule):
    """
    Ensure data preparation processes and caches the result.
    """
    assert os.path.exists(CACHE_PATH), "Cache file was not created"
    with open(CACHE_PATH, "rb") as f:
        cached_data = pickle.load(f)
    assert "init" in cached_data
    assert isinstance(cached_data["init"], torch_geometric.data.HeteroData)

def test_setup_creates_splits(datamodule):
    """
    Test that `setup()` correctly creates and stores train/val/test splits.
    """
    datamodule.setup()
    assert "train" in datamodule.data
    assert "val" in datamodule.data
    assert "test" in datamodule.data

def test_dataloaders_return_data(datamodule):
    """
    Ensure dataloaders return iterable data batches.
    """
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    assert train_batch is not None
    assert val_batch is not None
    assert test_batch is not None

def test_iterable_dataset(datamodule):
    """
    Test that iterable dataset returns expected tensors.
    """
    dataset = datamodule.get_iterable_dataset("disease")
    sample = next(iter(dataset))
    assert isinstance(sample, torch.Tensor)
