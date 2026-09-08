# Quantum Satellite Optimizer

A prototype for satellite constellation and orbit optimization using quantum computing, based on UC-096 of the QAIC 100 Quantum Use Cases document.

## Problem Statement

Efficiently assigning satellites to ground targets and optimizing their orbital parameters is a classical constrained combinatorial optimization problem. This project explores whether quantum algorithms (specifically QAOA - Quantum Approximate Optimization Algorithm) can formulate and solve simplified versions of this problem, and compares performance against classical optimization baselines.

## Objective

- Formulate satellite-target assignment as a constrained optimization problem (QUBO)
- Solve using QAOA with Qiskit/Aer simulation
- Compare results against classical optimization methods (OR-Tools, SciPy)
- Validate quantum solutions rigorously
- Visualize results through an interactive Streamlit dashboard

## Technology Stack

- **Quantum Computing**: Qiskit, Qiskit Aer (simulator)
- **Classical Optimization**: OR-Tools, SciPy
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly, Matplotlib
- **Web Framework**: Streamlit
- **Testing**: pytest

## Project Architecture

```
quantum-satellite-optimizer/
├── data/
│   ├── satellites.csv
│   └── targets.csv
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   ├── classical/
│   │   └── classical_optimizer.py
│   ├── quantum/
│   │   ├── qubo_model.py
│   │   └── qaoa_solver.py
│   ├── evaluation/
│   │   ├── comparison.py
│   │   └── validator.py
│   └── visualization/
│       └── plots.py
├── app/
│   └── app.py
├── tests/
│   ├── test_data.py
│   ├── test_classical.py
│   ├── test_qubo.py
│   └── test_validator.py
├── docs/
│   ├── methodology.md
│   └── architecture.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Development Workflow

1. **Data Preparation**: Load and preprocess synthetic satellite and target data
2. **Problem Formulation**: Convert assignment problem into QUBO format
3. **Classical Solution**: Solve using classical optimization (baseline)
4. **Quantum Solution**: Solve using QAOA with Qiskit Aer
5. **Validation**: Verify quantum solutions satisfy constraints
6. **Comparison**: Analyze solution quality, convergence, and computational metrics
7. **Visualization**: Display results through Streamlit dashboard

## Development Rules

- Keep the project modular and maintain separation of concerns
- Keep optimization logic separate from Streamlit UI
- Do not mix classical and quantum implementations
- Use synthetic data initially (2 satellites, 3 targets → 5 satellites, 10 targets)
- Validate every solution decoded from quantum circuits
- Follow current Qiskit APIs, avoid deprecated patterns
- Use same objective and constraints for fair comparison
- Add tests for mathematical and optimization functions
- Do not hard-code results or claim quantum advantage without evidence

## Current Status

**Phase 1: Project Setup** (In Progress)
- ✅ Repository initialized
- ✅ `.gitignore` configured
- ✅ `requirements.txt` created
- ✅ Project structure scaffolded
- ⏳ Awaiting implementation tasks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Coming soon. Implementation to follow.

## Testing

```bash
pytest tests/
```

## References

- QAIC 100 Quantum Use Cases - UC-096: Satellite Constellation Optimization
- Qiskit Documentation: https://qiskit.org/documentation/
- OR-Tools Documentation: https://developers.google.com/optimization

## License

MIT

## Contributors

- Mohammad Arshya Zaheen
