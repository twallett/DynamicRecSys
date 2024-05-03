# A Benchmark for Graph-Based Dynamic Recommendation Systems

## Overview

This repository contains resources related to the research paper "A Benchmark for Graph-Based Dynamic Recommendation Systems". The paper aims to provide a comprehensive benchmarking framework for evaluating the performance of graph-based dynamic recommendation systems.

## Folder Structure

```bash
.
├── README.md
├── code
│   ├── CaseStudy
│   └── RecSys
├── full_report
│   ├── FullReport.pdf
├── presentation
│   ├── GNN
│   ├── Presentation.pdf
│   └── Presentation.tex
└── research_paper
    ├── ResearchPaper.pdf 
    └── ResearchPaper.tex
```

## Requirements

Please pip install the following to run the code:

```bash
!pip install torch

!pip install torch_geometric

import torch

!pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
!pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install git+https://github.com/pyg-team/pytorch_geometric.git

!pip install faiss-gpu

!pip install cmake

!pip install git+https://github.com/pyg-team/pyg-lib.git

```

## Citation
If you find this work useful in your research, please cite the corresponding research paper:

```bash
TBD BibTex 
```

For any inquiries or feedback, please contact twallett@gwu.edu
