# Boolean Matrix Factorization (BMF)

## 📚 Overview

Boolean Matrix Factorization decomposes a boolean matrix **A** into two boolean matrices **B** and **C** such that **A ≈ B ⊙ C**. (where ⊙ is boolean product).

## 📦 Installation

### Requirements
- Python 3.7+

### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/tadtd/boolean-matrix-factorization.git
cd boolean-matrix-factorization

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate # on linux/macOS
# or
./venv/Scripts/activate # on Windows

# 4. Install dependencies
pip install -r requirements.txt
```

## 🧮 Available Algorithms

### Asso (Association-based BMF)
Uses association rules and greedy selection for factorization.

**Parameters:**
- `rank`: Number of factors (default: 5)
- `tau`: Confidence threshold (default: 0.8)
- `wp`: Weight for positive matches (default: 1.0)
- `wn`: Weight for negative penalty (default: 1.0)

### GreCon (Greedy Concept-based BMF)
Uses formal concept analysis and set covering approach.

**Parameters:** None required

### GreConD (Greedy Concept with Decomposition)
Enhanced version of GreCon with decomposition strategies.

**Parameters:** None required

## 🔗 Related Components

- **Set Covering**: Related optimization problem
- **Biclique Covering on bipartite graphs**: Graph-based problem variant
- **LDPC**: Error correction code applications

---

For detailed code examples and implementation details, refer to test files and algorithm implementations.
