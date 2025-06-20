"""
A test markdown class for pytest unit testing.
"""

import os
from vpeleaderboard.data.src.sbml.basico_model import BasicoModel
from app.data.markdown_report import generate_markdown_report

MODEL_PATH = "vpeleaderboard/data/src/sbml/models/BIOMD0000000064_url.xml"
def test_validate_sbml_file_path_success():
    """
    Test SBML directory validation when XML files exist.
    """
    assert os.path.exists("vpeleaderboard/data/src/sbml/models")
    assert len([f for f in os.listdir("vpeleaderboard/data/src/sbml/models") if f.endswith(".xml")]) > 0
    sbml_file_path = os.path.abspath(MODEL_PATH)

    model = BasicoModel(sbml_file_path=sbml_file_path)
    assert model is not None

def test_get_model_metadata():
    """
    Test the get_model_metadata method of the BasicoModel class.
    """
    model = BasicoModel(sbml_file_path=MODEL_PATH)
    metadata = model.get_model_metadata()
    assert metadata["Model Name"] is not None
    assert metadata["Number of Species"] >= 0
    assert metadata["Number of Parameters"] >= 0
    assert metadata["Description"] is not None

def test_generate_markdown_report():
    """
    Test the generate_markdown_report function.
    """
    folder_path = "vpeleaderboard/data/src/sbml/models"
    # folder_path = os.path.abspath(folder_path)
    output_path = "tests/test_index.md"

    template_dir = "app/templates"
    template_file = "data.html"

    assert os.path.exists(folder_path)
    generate_markdown_report(folder_path, template_dir, template_file, output_path)
    assert os.path.exists(output_path)

    os.remove(output_path)
