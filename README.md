# Assignment 2.1: Compute Blood Types

Bayesian Network solver for computing posterior ABO blood type distributions given a family tree, country allele priors, and test results (`bloodtype-test`, `mixed-bloodtype-test`, `pair-bloodtype-test`).

## Dependencies

- **Python**: 3.9.6
- **NumPy**: `numpy`

Install dependencies via pip:
```bash
pip install numpy
```

## How to Run

### Solve a directory of problems
```bash
python3 solver.py problems solutions
```
This reads all `problem-*.json` files in `problems/` and writes corresponding `solution-*.json` files into `solutions/`.

### Solve a single problem file
```bash
python3 solver.py problems/problem-a-00.json solutions/solution-a-00.json
```

### Verify against example solutions
```bash
python3 solver.py example-problems /tmp/my-example-solutions
python3 solution_check.py example-solutions /tmp/my-example-solutions
```

## Repository Structure

```
.
├── solver.py              # Bayesian network modeling & Variable Elimination solver
├── solution_check.py      # Provided script to compare solutions
├── SOLUTION_SUMMARY.md    # Description of the modeling & inference approach
├── README.md              # This file
├── example-problems/      # 80 example problem files
├── example-solutions/     # 80 reference solutions for example problems
├── problems/              # 80 assignment problem files
└── solutions/             # 80 generated solutions for problems
```
