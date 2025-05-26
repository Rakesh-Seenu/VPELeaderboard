# """
# A test markdown class for pytest unit testing with mocked BioBridgePrimeKG data.
# """

# import os
# import tempfile
# import pytest
# import pandas as pd
# from omegaconf import DictConfig
# from unittest.mock import MagicMock
# from app.data import kg_report

# @pytest.fixture
# def mock_cfg():
#     return DictConfig({
#         "data": {
#             "primekg_dir": "/mock/primekg",
#             "biobridge_dir": "/mock/biobridge",
#             "random_seed": 42,
#             "n_neg_samples": 5
#         }
#     })

# def test_get_kg_stats_success(monkeypatch, mock_cfg):
#     # Create mock triplets DataFrame
#     triplets = pd.DataFrame({
#         "head_index": [0, 1, 2],
#         "tail_index": [1, 2, 3],
#         "relation": ["r1", "r2", "r3"]
#     })

#     # Create mock BioBridgePrimeKG class
#     class MockKG:
#         def load_data(self): pass
#         def get_primekg_triplets(self): return triplets

#     monkeypatch.setattr(kg_report, "BioBridgePrimeKG", lambda cfg: MockKG())

#     num_nodes, num_edges = kg_report.get_kg_stats(mock_cfg)

#     assert num_nodes == 4  # unique nodes: 0,1,2,3
#     assert num_edges == 3

# def mock_create_markdown(data, template_dir, template_file):
#     mock_create_called_with.update({
#         "data": data,
#         "template_dir": template_dir,
#         "template_file": template_file
#     })
#     return "mocked markdown content"

# def mock_save_markdown(content, path):
#     mock_save_called_with.update({
#         "content": content,
#         "path": path
#     })
#     with open(path, "w") as f:
#         f.write(content)

# def test_generate_kg_markdown_report_with_models(monkeypatch):
#     with tempfile.TemporaryDirectory() as tmpdir:
#         template_file = "template.md"
#         output_file = os.path.join(tmpdir, "output.md")
#         template_path = os.path.join(tmpdir, template_file)

#         with open(template_path, "w") as f:
#             f.write("Dummy template content")

#         monkeypatch.setattr(kg_report, "create_markdown", mock_create_markdown)
#         monkeypatch.setattr(kg_report, "save_markdown", mock_save_markdown)

#         path_info = {
#             "template_dir": tmpdir,
#             "template_file": template_file,
#             "output_file": output_file
#         }

#         kg_report.generate_kg_markdown_report(
#             num_nodes=100,
#             num_edges=200,
#             models=[{"name": "ModelA", "abstract": "Test model"}],
#             path_info=path_info
#         )

#         assert mock_create_called_with["data"][0]["model_name"] == "ModelA"
#         assert os.path.exists(output_file)
#         assert mock_save_called_with["path"] == output_file
#         assert mock_save_called_with["content"] == "mocked markdown content"

#         # Cleanup mocks
#         mock_create_called_with.clear()
#         mock_save_called_with.clear()
"""
A test markdown class for pytest unit testing with mocked BioBridgePrimeKG data.
"""

import os
import tempfile
import pytest
import pandas as pd
from omegaconf import DictConfig
from unittest.mock import MagicMock
from app.data import kg_report

# Globals for mock tracking
mock_create_called_with = {}
mock_save_called_with = {}

@pytest.fixture
def mock_cfg():
    return DictConfig({
        "data": {
            "primekg_dir": "/mock/primekg",
            "biobridge_dir": "/mock/biobridge",
            "random_seed": 42,
            "n_neg_samples": 5,
            "models": [{"name": "MockModel", "abstract": "Some abstract"}],
            "kg_paths": {
                "template_dir": "/mock/templates",
                "template_file": "template.md",
                "output_file": "/mock/output.md"
            }
        }
    })


def test_get_kg_stats_success(monkeypatch, mock_cfg):
    # Create mock triplets DataFrame
    triplets = pd.DataFrame({
        "head_index": [0, 1, 2],
        "tail_index": [1, 2, 3],
        "relation": ["r1", "r2", "r3"]
    })

    # Mock BioBridgePrimeKG
    class MockKG:
        def load_data(self): pass
        def get_primekg_triplets(self): return triplets

    monkeypatch.setattr(kg_report, "BioBridgePrimeKG", lambda cfg: MockKG())

    num_nodes, num_edges = kg_report.get_kg_stats(mock_cfg)
    assert num_nodes == 4
    assert num_edges == 3


def test_get_kg_stats_failure(monkeypatch, mock_cfg):
    # Raise ValueError in get_primekg_triplets
    class MockKG:
        def load_data(self): pass
        def get_primekg_triplets(self): raise ValueError("Mock error")

    monkeypatch.setattr(kg_report, "BioBridgePrimeKG", lambda cfg: MockKG())

    num_nodes, num_edges = kg_report.get_kg_stats(mock_cfg)
    assert num_nodes == 0
    assert num_edges == 0


def mock_create_markdown(data, template_dir, template_file):
    mock_create_called_with.update({
        "data": data,
        "template_dir": template_dir,
        "template_file": template_file
    })
    return "mocked markdown content"


def mock_save_markdown(content, path):
    mock_save_called_with.update({
        "content": content,
        "path": path
    })
    with open(path, "w") as f:
        f.write(content)


def test_generate_kg_markdown_report_with_models(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        template_file = "template.md"
        output_file = os.path.join(tmpdir, "output.md")
        template_path = os.path.join(tmpdir, template_file)

        with open(template_path, "w") as f:
            f.write("Dummy template content")

        monkeypatch.setattr(kg_report, "create_markdown", mock_create_markdown)
        monkeypatch.setattr(kg_report, "save_markdown", mock_save_markdown)

        path_info = {
            "template_dir": tmpdir,
            "template_file": template_file,
            "output_file": output_file
        }

        kg_report.generate_kg_markdown_report(
            num_nodes=100,
            num_edges=200,
            models=[{"name": "ModelA", "abstract": "Test model"}],
            path_info=path_info
        )

        assert mock_create_called_with["data"][0]["model_name"] == "ModelA"
        assert os.path.exists(output_file)
        assert mock_save_called_with["path"] == output_file
        assert mock_save_called_with["content"] == "mocked markdown content"

        mock_create_called_with.clear()
        mock_save_called_with.clear()


def test_generate_kg_markdown_report_without_models(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        template_file = "template.md"
        output_file = os.path.join(tmpdir, "output.md")
        template_path = os.path.join(tmpdir, template_file)

        with open(template_path, "w") as f:
            f.write("Dummy template content")

        monkeypatch.setattr(kg_report, "create_markdown", mock_create_markdown)
        monkeypatch.setattr(kg_report, "save_markdown", mock_save_markdown)

        path_info = {
            "template_dir": tmpdir,
            "template_file": template_file,
            "output_file": output_file
        }

        kg_report.generate_kg_markdown_report(
            num_nodes=100,
            num_edges=200,
            models=[],  # Empty list triggers else block
            path_info=path_info
        )

        assert os.path.exists(output_file)
        assert mock_save_called_with["content"] == "mocked markdown content"

        mock_create_called_with.clear()
        mock_save_called_with.clear()


def test_main_function(monkeypatch):
    # Create fake config
    fake_config = DictConfig({
        "data": {
            "models": [{"name": "MainModel", "abstract": "Main run"}],
            "kg_paths": {
                "template_dir": "/fake",
                "template_file": "template.md",
                "output_file": "out.md"
            }
        }
    })

    # Mock dependencies
    mock_get_stats = MagicMock(return_value=(111, 222))
    mock_generate = MagicMock()

    monkeypatch.setattr(kg_report, "get_kg_stats", mock_get_stats)
    monkeypatch.setattr(kg_report, "generate_kg_markdown_report", mock_generate)

    kg_report.main(fake_config)

    mock_get_stats.assert_called_once_with(fake_config)
    mock_generate.assert_called_once_with(
        num_nodes=111,
        num_edges=222,
        models=fake_config.data.models,
        path_info=fake_config.data.kg_paths
    )
