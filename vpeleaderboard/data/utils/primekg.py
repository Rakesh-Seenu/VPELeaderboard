"""
Test cases for datasets/primekg_loader.py
"""

import os
from omegaconf import DictConfig
import requests
from tqdm import tqdm
import pandas as pd

class PrimeKG:
    """
    Class for loading PrimeKG dataset.
    It downloads the data from the Harvard Dataverse and stores it in the local directory.
    The data is then loaded into pandas DataFrame of nodes and edges.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Constructor for PrimeKG class.

        Args:
            local_dir (str): The local directory where the data will be stored.
        """
        self.name: str = "primekg"
        self.server_path: str = "https://dataverse.harvard.edu/api/access/datafile/"
        self.file_ids: dict = {"nodes": 6180617, "edges": 6180616}

        if isinstance(cfg, DictConfig):
            self.local_dir: str = cfg.data.primekg_dir
        elif isinstance(cfg, dict):
            self.local_dir: str = cfg["data"]["primekg_dir"]
        elif isinstance(cfg, str):
            self.local_dir: str = cfg
        else:
            raise TypeError(f"Unsupported config type: {type(cfg)}")

        # Attributes to store the data
        self.nodes: pd.DataFrame = None
        self.edges: pd.DataFrame = None
        os.makedirs(os.path.dirname(self.local_dir), exist_ok=True)


    def _download_file(self, remote_url:str, local_path: str):
        """
        A helper function to download a file from remote URL to the local directory.

        Args:
            remote_url (str): The remote URL of the file to be downloaded.
            local_path (str): The local path where the file will be saved.
        """
        response = requests.get(remote_url, stream=True, timeout=300)
        response.raise_for_status()
        progress_bar = tqdm(
            total=int(response.headers.get("content-length", 0)),
            unit="iB",
            unit_scale=True,
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as file:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    def _load_nodes(self) -> pd.DataFrame:
        """
        Private method to load the nodes dataframe of PrimeKG dataset.
        This method downloads the nodes file from the Harvard Dataverse if it does not exist
        in the local directory. Otherwise, it loads the data from the local directory.
        It further processes the dataframe of nodes and returns it.

        Returns:
            The nodes dataframe of PrimeKG dataset.
        """
        local_file = os.path.join(self.local_dir, f"{self.name}_nodes.tsv.gz")
        if os.path.exists(local_file):
            print(f"{local_file} already exists. Loading the data from the local directory.")
            nodes = pd.read_csv(local_file, sep="\t", compression="gzip", low_memory=False)
        else:
            print(f"Downloading node file from {self.server_path}{self.file_ids['nodes']}")
            self._download_file(f"{self.server_path}{self.file_ids['nodes']}",
                                os.path.join(self.local_dir, "nodes.tab"))
            nodes = pd.read_csv(os.path.join(self.local_dir, "nodes.tab"),
                                     sep="\t", low_memory=False)
            nodes = nodes[
                ["node_index", "node_name", "node_source", "node_id", "node_type"]
            ]
            nodes.to_csv(local_file, index=False, sep="\t", compression="gzip")

        return nodes

    def _load_edges(self, nodes: pd.DataFrame) -> pd.DataFrame:
        """
        Private method to load the edges dataframe of PrimeKG dataset.
        This method downloads the edges file from the Harvard Dataverse if it does not exist
        in the local directory. Otherwise, it loads the data from the local directory.
        It further processes the dataframe of edges and returns it.

        Args:
            nodes (pd.DataFrame): The nodes dataframe of PrimeKG dataset.

        Returns:
            The edges dataframe of PrimeKG dataset.
        """
        local_file = os.path.join(self.local_dir, f"{self.name}_edges.tsv.gz")
        if os.path.exists(local_file):
            print(f"{local_file} already exists. Loading the data from the local directory.")
            edges = pd.read_csv(local_file, sep="\t", compression="gzip", low_memory=False)
        else:
            print(f"Downloading edge file from {self.server_path}{self.file_ids['edges']}")
            self._download_file(f"{self.server_path}{self.file_ids['edges']}",
                                os.path.join(self.local_dir, "edges.csv"))
            edges = pd.read_csv(os.path.join(self.local_dir, "edges.csv"),
                                     sep=",", low_memory=False)
            edges = edges.merge(
                nodes, left_on="x_index", right_on="node_index"
            )
            edges.drop(["x_index"], axis=1, inplace=True)
            edges.rename(
                columns={
                    "node_index": "head_index",
                    "node_name": "head_name",
                    "node_source": "head_source",
                    "node_id": "head_id",
                    "node_type": "head_type",
                },
                inplace=True,
            )
            edges = edges.merge(
                nodes, left_on="y_index", right_on="node_index"
            )
            edges.drop(["y_index"], axis=1, inplace=True)
            edges.rename(
                columns={
                    "node_index": "tail_index",
                    "node_name": "tail_name",
                    "node_source": "tail_source",
                    "node_id": "tail_id",
                    "node_type": "tail_type"
                },
                inplace=True,
            )
            edges = edges[
                [
                    "head_index", "head_name", "head_source", "head_id", "head_type",
                    "tail_index", "tail_name", "tail_source", "tail_id", "tail_type",
                    "display_relation", "relation",
                ]
            ]
            edges.to_csv(local_file, index=False, sep="\t", compression="gzip")

        return edges

    def load_data(self):
        """
        Load the PrimeKG dataset into pandas DataFrame of nodes and edges.
        """
        self.nodes = self._load_nodes()
        self.edges = self._load_edges(self.nodes)

    def get_nodes(self) -> pd.DataFrame:
        """
        Get the nodes dataframe of PrimeKG dataset.

        Returns:
            The nodes dataframe of PrimeKG dataset.
        """
        return self.nodes

    def get_edges(self) -> pd.DataFrame:
        """
        Get the edges dataframe of PrimeKG dataset.

        Returns:
            The edges dataframe of PrimeKG dataset.
        """
        return self.edges
