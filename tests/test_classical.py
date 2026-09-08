"""
Tests for classical optimizer.

Verifies that the brute-force classical solver correctly finds the optimal
solution to the satellite-target assignment problem.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import load_problem
from src.classical.classical_optimizer import ClassicalOptimizer


@pytest.fixture
def problem_instance():
    """Load the toy problem instance."""
    problem_path = Path(__file__).parent.parent.parent / "data" / "sample_problem.json"
    return load_problem(str(problem_path))


@pytest.fixture
def optimizer(problem_instance):
    """Create optimizer instance."""
    return ClassicalOptimizer(problem_instance)


@pytest.fixture
def result(optimizer):
    """Solve the problem."""
    return optimizer.solve()


class TestClassicalOptimizer:
    """Test suite for classical optimizer."""
    
    def test_solver_returns_feasible_solution(self, result):
        """Test that solver finds a feasible solution."""
        assert result['feasible'], "Solver should find a feasible solution"
    
    def test_optimal_vector_exact(self, result):
        """Test that optimal vector is exactly [1, 1, 0, 1]."""
        expected = [1, 1, 0, 1]
        assert result['optimal_vector'] == expected, \
            f"Expected {expected}, got {result['optimal_vector']}"
    
    def test_variable_names_exact(self, result):
        """Test that variable names are exactly ['x11', 'x12', 'x21', 'x23']."""
        expected = ['x11', 'x12', 'x21', 'x23']
        assert result['variable_names'] == expected, \
            f"Expected {expected}, got {result['variable_names']}"
    
    def test_objective_value_exact(self, result):
        """Test that objective value is exactly 4.0."""
        expected = 4.0
        assert result['objective_value'] == expected, \
            f"Expected objective {expected}, got {result['objective_value']}"
    
    def test_all_targets_covered(self, problem_instance, result):
        """Test that all three targets are covered."""
        num_targets = problem_instance.get_num_targets()
        num_covered = len(result['target_coverage'])
        assert num_covered == num_targets, \
            f"Expected to cover {num_targets} targets, covered {num_covered}"
    
    def test_target_coverage_priorities(self, result):
        """Test that target priorities are correct."""
        expected_priorities = {1: 10, 2: 8, 3: 12}
        for tgt_id, (sat_id, priority) in result['target_coverage'].items():
            assert priority == expected_priorities[tgt_id], \
                f"Target {tgt_id}: expected priority {expected_priorities[tgt_id]}, got {priority}"
    
    def test_sat1_energy_usage(self, result):
        """Test that SAT-1 energy usage is exactly 27."""
        expected = 27.0
        actual = result['satellite_energy_usage'].get(1, 0.0)
        assert actual == expected, \
            f"SAT-1 energy: expected {expected}, got {actual}"
    
    def test_sat2_energy_usage(self, result):
        """Test that SAT-2 energy usage is exactly 25."""
        expected = 25.0
        actual = result['satellite_energy_usage'].get(2, 0.0)
        assert actual == expected, \
            f"SAT-2 energy: expected {expected}, got {actual}"
    
    def test_target_uniqueness_satisfied(self, result):
        """Test that target uniqueness constraint is satisfied."""
        # Check constraint_info
        assert result['constraint_info']['target_uniqueness'], \
            "Target uniqueness constraint violated"
        
        # Double-check: no target assigned to multiple satellites
        target_counts = {}
        for (sat_id, tgt_id), val in result['assignments'].items():
            if val == 1:
                target_counts[tgt_id] = target_counts.get(tgt_id, 0) + 1
        
        for tgt_id, count in target_counts.items():
            assert count <= 1, f"Target {tgt_id} assigned to {count} satellites"
    
    def test_satellite_capacity_satisfied(self, result):
        """Test that satellite capacity constraint is satisfied."""
        # Check constraint_info
        assert result['constraint_info']['satellite_capacity'], \
            "Satellite capacity constraint violated"
        
        # Double-check: each satellite not over capacity
        for sat_id, (num_assigned, capacity) in result['satellite_capacity_usage'].items():
            assert num_assigned <= capacity, \
                f"SAT-{sat_id}: assigned {num_assigned} targets, capacity {capacity}"
    
    def test_energy_budget_satisfied(self, result):
        """Test that energy budget constraint is satisfied."""
        # Check constraint_info
        assert result['constraint_info']['energy_budget'], \
            "Energy budget constraint violated"
        
        # Double-check: each satellite within budget
        for sat_id, energy in result['satellite_energy_usage'].items():
            budget = 100 if sat_id == 1 else 80
            assert energy <= budget, \
                f"SAT-{sat_id}: used {energy} energy, budget {budget}"
    
    def test_sat1_capacity_at_limit(self, result):
        """Test that SAT-1 capacity is fully utilized (2/2)."""
        num_assigned, capacity = result['satellite_capacity_usage'][1]
        assert num_assigned == 2 and capacity == 2, \
            f"SAT-1 capacity: expected 2/2, got {num_assigned}/{capacity}"
    
    def test_sat2_capacity_at_limit(self, result):
        """Test that SAT-2 capacity is fully utilized (1/1)."""
        num_assigned, capacity = result['satellite_capacity_usage'][2]
        assert num_assigned == 1 and capacity == 1, \
            f"SAT-2 capacity: expected 1/1, got {num_assigned}/{capacity}"
    
    def test_total_priority_covered(self, result):
        """Test that total priority covered is 30 (sum of 10+8+12)."""
        expected = 30
        actual = result['total_priority_covered']
        assert actual == expected, \
            f"Expected total priority {expected}, got {actual}"
    
    def test_brute_force_evaluates_all_combinations(self, optimizer, result):
        """Test that brute-force evaluates all 16 possible binary combinations."""
        total_evaluated = result['num_feasible'] + result['num_infeasible']
        expected_total = 2 ** optimizer.num_variables  # 2^4 = 16
        assert total_evaluated == expected_total, \
            f"Expected {expected_total} combinations evaluated, got {total_evaluated}"
    
    def test_at_least_one_feasible_solution(self, result):
        """Test that at least one feasible solution exists."""
        assert result['num_feasible'] >= 1, \
            f"Expected at least 1 feasible solution, found {result['num_feasible']}"
    
    def test_sat1_observes_t1_and_t2(self, result):
        """Test that SAT-1 is assigned to T1 and T2."""
        assignments = result['assignments']
        assert (1, 1) in assignments and assignments[(1, 1)] == 1, \
            "SAT-1 not assigned to T1"
        assert (1, 2) in assignments and assignments[(1, 2)] == 1, \
            "SAT-1 not assigned to T2"
    
    def test_sat2_observes_t3(self, result):
        """Test that SAT-2 is assigned to T3."""
        assignments = result['assignments']
        assert (2, 3) in assignments and assignments[(2, 3)] == 1, \
            "SAT-2 not assigned to T3"
    
    def test_sat2_not_observes_t1(self, result):
        """Test that SAT-2 is not assigned to T1."""
        assignments = result['assignments']
        assert (2, 1) not in assignments or assignments[(2, 1)] == 0, \
            "SAT-2 should not be assigned to T1"


class TestClassicalOptimizerProperties:
    """Property-based tests for the optimizer."""
    
    def test_objective_is_non_negative_for_all_feasible(self, optimizer):
        """Test that all feasible solutions have objective >= some minimum."""
        result = optimizer.solve()
        # For this problem, the optimal is 4.0
        # All other feasible solutions should have lower objective
        for eval_dict in optimizer.all_evaluations:
            if eval_dict['feasible']:
                assert eval_dict['objective'] <= result['objective_value'], \
                    "Found a feasible solution with better objective"
    
    def test_optimizer_summary_generation(self, optimizer, result):
        """Test that optimizer can generate a summary string."""
        summary = optimizer.get_summary(result)
        assert isinstance(summary, str), "Summary should be a string"
        assert 'CLASSICAL OPTIMIZER RESULT' in summary, "Summary should mention optimizer"
        assert 'FEASIBLE' in summary, "Summary should mention feasibility"
        assert str(result['objective_value']) in summary or '4.0' in summary, \
            "Summary should mention objective value"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
