# A Benchmark for Graph-Based Dynamic Recommendation Systems

## Overview

This repository contains resources related to the research paper "A Benchmark for Graph-Based Dynamic Recommendation Systems". The paper aims to provide a comprehensive benchmarking framework for evaluating the performance of graph-based dynamic recommendation systems.

## FAQ

### How to run code?

[click here](https://vimeo.com/942515870?share=copy)

### What is the folder structure?

```bash
.
├── README.md
├── README_files
│   └── libs
├── code
│   ├── CaseStudy
│   └── RecSys
├── full_report
│   ├── FinalReport.pdf
│   └── Report
├── presentation
│   ├── GNN
│   ├── Presentation.pdf
│   └── Presentation.tex
├── research_paper
│   ├── Paper
│   └── ResearchPaper.pdf
└── runme.py
```

### What Python libraries are required?

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

### What citation should I use?

If you find this work useful in your research, please cite the corresponding research paper:

```bash
TBD BibTex 
```

For any inquiries or feedback, please contact twallett@gwu.edu
