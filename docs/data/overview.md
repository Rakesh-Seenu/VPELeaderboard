---
hide:
  - navigation
---

# 🧬 Data Overview

Welcome to the **Data Section** of the [VPE Leaderboard](https://github.com/VirtualPatientEngine) platform. This section provides the tools, documentation, and loaders needed to work with two primary data formats: **[SBML models](index.md)** and **[Knowledge Graphs (KGs)](index_kg.md)**.

Choose the workflow relevant to your data:

<div class="grid cards" markdown>

-   :material-flask-outline:{ .lg } **SBML Dataloader**

    ---

    Parse and load biological models in **SBML (Systems Biology Markup Language)** format. This loader extracts species, reactions, and parameters, preparing them for simulation and analysis.

    [📄 Code Documentation](sbml_dataloader.md){ .md-button .md-button}     [📘 Tutorial Notebook](../notebooks/sbml_dataloader.ipynb){ .md-button .md-button }


-   :material-brain:{ .lg } **Knowledge Graph Dataloader**

    ---

    Import and standardize structured graph datasets such as biomedical or molecular knowledge graphs. Includes preprocessing, normalization, and integration into our evaluation pipelines.

    [📄 Code Documentation](kg_dataloader.md){ .md-button .md-button}     [📘 Tutorial Notebook](../notebooks/KG_dataloader.ipynb){ .md-button .md-button}

</div>

---

If you're unsure which data format your model fits into, we recommend reviewing both loaders above and their respective notebooks before proceeding. For further questions, reach out to the development team or file an issue on the [GitHub repository](https://github.com/VirtualPatientEngine).
