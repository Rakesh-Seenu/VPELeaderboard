"""
Markdown report generator for BioBridge PrimeKG statistics and model metadata.

This script loads data using the BioBridgePrimeKG class,
extracts knowledge graph statistics, and generates a markdown report.
"""

import os
import sys
import logging
import pandas as pd
import hydra
from omegaconf import DictConfig
from vpeleaderboard.data.src.kg.biobridge_primekg import BioBridgePrimeKG
from app.utils import create_markdown, save_markdown

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def get_kg_stats(config: DictConfig) -> tuple[int, int]:
    """
    Compute number of nodes and edges from the BioBridgePrimeKG dataset.

    Args:
        config (DictConfig): Configuration object.

    Returns:
        tuple[int, int]: Number of nodes, number of edges.
    """
    try:
        biobridge_dataset = BioBridgePrimeKG(config)
        biobridge_dataset.load_data()
        triplets = biobridge_dataset.get_primekg_triplets()

        num_edges = triplets.shape[0]
        all_node_indices = pd.concat([triplets['head_index'], triplets['tail_index']]).unique()
        num_nodes = len(all_node_indices)

        logger.info("KG Stats — Nodes: %d, Edges: %d", num_nodes, num_edges)
        return num_nodes, num_edges
    except ValueError as e:
        logger.error("Error while computing KG stats: %s", str(e))
        return 0, 0
def generate_kg_markdown_report(
    num_nodes: int,
    num_edges: int,
    models: list,
    path_info: dict # Combined template_dir, template_file, output_file
) -> None:
    """
    Generate and save a markdown report for KG statistics and model metadata.

    Args:
        num_nodes (int): Total number of nodes in the KG.
        num_edges (int): Total number of edges in the KG.
        models (list): List of model metadata dictionaries.
        path_info (dict): Dictionary containing 'template_dir', 'template_file', and 'output_file'.
    """
    table_data = []

    # Extract paths from path_info
    template_dir = path_info.get("template_dir")
    template_file = path_info.get("template_file")
    output_file = path_info.get("output_file")

    if models:
        for model in models:
            table_data.append({
                "model_name": model.get("name", "N/A"),
                "abstract": model.get("abstract", "N/A"),
                "num_nodes": num_nodes,
                "num_edges": num_edges
            })
    else:
        logger.info("No model metadata provided; only KG stats will be included.")
    metadata_df = pd.DataFrame(table_data)
    metadata_records = metadata_df.to_dict(orient='records')
    markdown_content = create_markdown(metadata_records, template_dir, template_file)
    save_markdown(markdown_content, output_file)
    logger.info("Markdown report saved at: %s", output_file)


def main(config: DictConfig):
    """
    Main function to orchestrate the report generation.

    Args:
        config (DictConfig): Hydra configuration object.
    """
    logger.info("Starting KG markdown report generation...")

    models = config.data.get("models", [])
    kg_paths = config.data.get("kg_paths", {})
    logger.info("Loaded kg_paths from config: %s", kg_paths)

    num_nodes, num_edges = get_kg_stats(config)

    # Pass kg_paths directly as path_info
    generate_kg_markdown_report(
        num_nodes=num_nodes,
        num_edges=num_edges,
        models=models,
        path_info=kg_paths # Now passing a single dict
    )


if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg_loaded = hydra.compose(config_name="config", overrides=["data=default"])
    main(cfg_loaded)
