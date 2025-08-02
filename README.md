# Boolean Matrix Factorization (BMF)

A Python-based project for studying and experimenting with Boolean Matrix Factorization. It provides a modular framework for exploring and comparing different BMF algorithms.

In addition, this repository aims to reproduce the results of the paper "Boolean Matrix Factorization and Noisy Completion via Message Passing" by \textbf{Siamak Ravanbakhsh, Barnabás Póczos, and Russell Greiner.}

## Overview

Boolean Matrix Factorization decomposes a binary matrix **A** into the Boolean product of two factor matrices **B** and **C**, such that **A ≈ B ∘ C**, where ∘ denotes the Boolean matrix product.

### Applications

BMF is closely related to important combinatorial optimization problems:
- **Set Covering Problem**: BMF can be viewed as finding a minimal set of basis vectors that cover all data points, where each factor represents a subset of features
- **Biclique Covering Problem**: Finding a minimal collection of bicliques (complete bipartite subgraphs) that cover all edges in a bipartite graph, directly corresponding to Boolean matrix factorization

This library currently implements:
- **Greedy BMF**: Greedy rank-1 approximation approach
- **ASSO Algorithm**: Association-based factorization using similarity thresholds

## Installation

### Requirements
- Python ≥ 3.10
- NumPy ≥ 1.15.4

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
BMF/                          # Main library package
├── __init__.py              # Package initialization
├── utils.py                 # Utility functions and BMFResult class
└── models/                  # Algorithm implementations
    ├── __init__.py
    ├── base.py             # Abstract base class
    ├── asso.py             # ASSO algorithm
    └── greedy.py           # Greedy BMF algorithm

problems/                    # Related optimization problems
├── set-covering/           # Set covering problem implementations
│   ├── greedy.py          # Greedy set covering
│   ├── lp.py              # Linear programming approach
│   └── main.py
└── LDPC/                   # LDPC codes (related research)
    ├── LDPC.py            # LDPC implementation
    ├── main.py
    ├── environment.yml    # Conda environment
    └── README.md

test.py                     # Example usage and testing
requirements.txt            # Python dependencies
LICENSE                     
```

## Contributing

Contributions are welcome! If you'd like to contribute to this project:

1. **Issues**: Feel free to open an issue to report bugs, request features, or discuss improvements
2. **Pull Requests**: Submit a pull request with your changes.
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is intended for research and educational purposes.

## Contact

For research collaboration or questions about implementation details, feel free to contact me via email: [dotiendat1725@gmail.com](mailto:dotiendat1725@gmail.com)

---

*This library serves as a foundation for Boolean Matrix Factorization research and can be extended for various applications in data mining, pattern recognition, and optimization.*
