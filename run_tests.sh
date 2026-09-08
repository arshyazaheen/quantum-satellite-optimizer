#!/bin/bash
# Test runner script for Quantum Satellite Optimizer
# This script runs pytest with quiet output on the test suite

cd "$(dirname "$0")" || exit 1

echo "Running pytest for Quantum Satellite Optimizer..."
echo "================================================="
echo ""

# Run pytest with quiet flag (-q) to show minimal output
python -m pytest tests/ -q --tb=short

echo ""
echo "================================================="
echo "Test run complete"
