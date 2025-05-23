"""
Test cases for datasets/primekg_loader.py
"""

import os
import shutil
import pytest
import tempfile
from omegaconf import OmegaConf
from vpeleaderboard.data.utils.primekg import PrimeKG

@pytest.fixture(scope="module")
def primekg_config():
    """
    Create a fake Hydra config for PrimeKG using temp directories.
    """
    tmpdir = tempfile.mkdtemp()

    config = OmegaConf.create({
        "data": {
            "primekg_dir": os.path.join(tmpdir, "primekg")
        }
    })

    # Clean up the temporary directory after tests are done
    yield config
    # shutil.rmtree(tmpdir)

@pytest.fixture(scope="module")
def primekg_instance(primekg_config):
    """
    Fixture for instantiating PrimeKG.
    """
    primekg_dir = primekg_config.data.primekg_dir

    # Ensure the directory exists before PrimeKG tries to use it
    os.makedirs(primekg_dir, exist_ok=True)

    instance = PrimeKG(primekg_dir)
    return instance    

def test_download_primekg(primekg_instance):
    """
    Test the loading method of the PrimeKG class by downloading PrimeKG from server.
    """
    
    # Load PrimeKG data
    primekg_instance.load_data()
    primekg_nodes = primekg_instance.get_nodes()
    primekg_edges = primekg_instance.get_edges()

    # Check if the local directory exists
    assert os.path.exists(primekg_instance.local_dir)
    # Check if downloaded and processed files exist
    files = ["nodes.tab", f"{primekg_instance.name}_nodes.tsv.gz",
             "edges.csv", f"{primekg_instance.name}_edges.tsv.gz"]
    for file in files:
        path = f"{primekg_instance.local_dir}/{file}"
        assert os.path.exists(path)
    # Check processed PrimeKG dataframes
    # Nodes
    assert primekg_nodes is not None
    assert len(primekg_nodes) > 0
    assert primekg_nodes.shape[0] == 129375
    # Edges
    assert primekg_edges is not None
    assert len(primekg_edges) > 0
    assert primekg_edges.shape[0] == 8100498

def test_load_existing_primekg(primekg_instance):
    """
    Test the loading method of the PrimeKG class by loading existing PrimeKG in local.
    """
    # Load PrimeKG data
    primekg_instance.load_data()
    primekg_nodes = primekg_instance.get_nodes()
    primekg_edges = primekg_instance.get_edges()

    # Check if the local directory exists
    assert os.path.exists(primekg_instance.local_dir)
    # Check if downloaded and processed files exist
    files = ["nodes.tab", f"{primekg_instance.name}_nodes.tsv.gz",
             "edges.csv", f"{primekg_instance.name}_edges.tsv.gz"]
    for file in files:
        path = f"{primekg_instance.local_dir}/{file}"
        assert os.path.exists(path)
    # Check processed PrimeKG dataframes
    # Nodes
    assert primekg_nodes is not None
    assert len(primekg_nodes) > 0
    assert primekg_nodes.shape[0] == 129375
    # Edges
    assert primekg_edges is not None
    assert len(primekg_edges) > 0
    assert primekg_edges.shape[0] == 8100498