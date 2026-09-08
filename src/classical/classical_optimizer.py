"""
Classical exact optimizer for satellite-target assignment problem.

Uses brute-force enumeration of all binary combinations to find the optimal
solution to the constrained maximization problem.
"""

from typing import Dict, List, Tuple, Optional
import itertools
import numpy as np
from src.data.preprocessing import ProblemInstance


class ClassicalOptimizer:
    """
    Classical exact optimizer for satellite-target assignment.
    
    Enumerates all possible binary assignments and evaluates feasibility
    and objective value to find the optimal solution.
    """
    
    def __init__(self, problem: ProblemInstance):
        """
        Initialize optimizer with a problem instance.
        
        Args:
            problem: ProblemInstance object with satellites, targets, costs, etc.
        """
        self.problem = problem
        self.variable_names = problem.get_variable_names()
        self.num_variables = problem.get_num_variables()
        self.feasible_pairs = problem.get_feasible_pairs()
        
        # Extract objective weights
        obj_weights = problem.get_objective_weights()
        self.w_priority = obj_weights['w_priority']
        self.w_cost = obj_weights['w_cost']
        
        # Store feasible pairs as dict for quick lookup
        self.pair_index = {pair: idx for idx, pair in enumerate(self.feasible_pairs)}
        
        # Results storage
        self.optimal_vector = None
        self.optimal_value = None
        self.num_feasible = 0
        self.num_infeasible = 0
        self.all_evaluations = []
    
    def _get_variable_index(self, sat_id: int, tgt_id: int) -> Optional[int]:
        """
        Get index of variable x_ij in the feasible pairs list.
        
        Args:
            sat_id: Satellite ID.
            tgt_id: Target ID.
            
        Returns:
            Index if pair is feasible, None otherwise.
        """
        pair = (sat_id, tgt_id)
        return self.pair_index.get(pair, None)
    
    def _vector_to_assignment(self, binary_vector: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Convert binary vector to assignment dictionary.
        
        Args:
            binary_vector: List of binary values in feasible pair order.
            
        Returns:
            Dict mapping (sat_id, tgt_id) -> assignment (0 or 1).
        """
        assignment = {}
        for idx, (sat_id, tgt_id) in enumerate(self.feasible_pairs):
            assignment[(sat_id, tgt_id)] = binary_vector[idx]
        return assignment
    
    def _calculate_objective(self, binary_vector: List[int]) -> float:
        """
        Calculate objective value for a binary assignment vector.
        
        Objective: Maximize Z = w_p * sum(priority_j * x_ij) - w_c * sum(cost_ij * x_ij)
        
        Args:
            binary_vector: List of binary values for each feasible pair.
            
        Returns:
            Objective value (higher is better).
        """
        obj = 0.0
        
        for idx, (sat_id, tgt_id) in enumerate(self.feasible_pairs):
            if binary_vector[idx] == 1:
                priority = self.problem.get_target_priority(tgt_id)
                cost = self.problem.get_observation_cost(sat_id, tgt_id)
                
                obj += self.w_priority * priority - self.w_cost * cost
        
        return obj
    
    def _check_target_uniqueness(self, binary_vector: List[int]) -> Tuple[bool, Dict[int, int]]:
        """
        Check target assignment uniqueness constraint: each target assigned to at most 1 satellite.
        
        Args:
            binary_vector: List of binary values.
            
        Returns:
            (is_feasible, target_assignment_counts) tuple.
        """
        target_counts = {}
        
        for idx, (sat_id, tgt_id) in enumerate(self.feasible_pairs):
            if binary_vector[idx] == 1:
                if tgt_id not in target_counts:
                    target_counts[tgt_id] = 0
                target_counts[tgt_id] += 1
        
        # Check: each target has at most 1 assignment
        for tgt_id in self.problem.get_all_target_ids():
            count = target_counts.get(tgt_id, 0)
            if count > 1:
                return False, target_counts
        
        return True, target_counts
    
    def _check_satellite_capacity(self, binary_vector: List[int]) -> Tuple[bool, Dict[int, int]]:
        """
        Check satellite capacity constraint: each satellite observes at most capacity targets.
        
        Args:
            binary_vector: List of binary values.
            
        Returns:
            (is_feasible, satellite_assignment_counts) tuple.
        """
        sat_counts = {}
        
        for idx, (sat_id, tgt_id) in enumerate(self.feasible_pairs):
            if binary_vector[idx] == 1:
                if sat_id not in sat_counts:
                    sat_counts[sat_id] = 0
                sat_counts[sat_id] += 1
        
        # Check: each satellite has at most capacity assignments
        for sat_id in self.problem.get_all_satellite_ids():
            count = sat_counts.get(sat_id, 0)
            capacity = self.problem.get_satellite_capacity(sat_id)
            if count > capacity:
                return False, sat_counts
        
        return True, sat_counts
    
    def _check_energy_budget(self, binary_vector: List[int]) -> Tuple[bool, Dict[int, float]]:
        """
        Check satellite energy budget constraint.
        
        Args:
            binary_vector: List of binary values.
            
        Returns:
            (is_feasible, satellite_energy_usage) tuple.
        """
        sat_energy = {}
        
        for idx, (sat_id, tgt_id) in enumerate(self.feasible_pairs):
            if binary_vector[idx] == 1:
                cost = self.problem.get_observation_cost(sat_id, tgt_id)
                if sat_id not in sat_energy:
                    sat_energy[sat_id] = 0.0
                sat_energy[sat_id] += cost
        
        # Check: each satellite energy usage <= budget
        for sat_id in self.problem.get_all_satellite_ids():
            usage = sat_energy.get(sat_id, 0.0)
            budget = self.problem.get_satellite_energy_budget(sat_id)
            if usage > budget:
                return False, sat_energy
        
        return True, sat_energy
    
    def _is_feasible(self, binary_vector: List[int]) -> Tuple[bool, Dict]:
        """
        Check if binary vector satisfies all constraints.
        
        Args:
            binary_vector: List of binary values.
            
        Returns:
            (is_feasible, constraint_info) tuple with detailed constraint status.
        """
        constraint_info = {}
        
        # Check target uniqueness
        target_unique, target_counts = self._check_target_uniqueness(binary_vector)
        constraint_info['target_uniqueness'] = target_unique
        constraint_info['target_counts'] = target_counts
        
        if not target_unique:
            return False, constraint_info
        
        # Check satellite capacity
        sat_capacity, sat_counts = self._check_satellite_capacity(binary_vector)
        constraint_info['satellite_capacity'] = sat_capacity
        constraint_info['satellite_counts'] = sat_counts
        
        if not sat_capacity:
            return False, constraint_info
        
        # Check energy budget
        energy_ok, sat_energy = self._check_energy_budget(binary_vector)
        constraint_info['energy_budget'] = energy_ok
        constraint_info['satellite_energy'] = sat_energy
        
        if not energy_ok:
            return False, constraint_info
        
        return True, constraint_info
    
    def _get_target_coverage(self, binary_vector: List[int]) -> Dict[int, Tuple[int, float]]:
        """
        Get target coverage information: which targets are assigned and their priorities.
        
        Args:
            binary_vector: List of binary values.
            
        Returns:
            Dict mapping target_id -> (assigned_satellite_id, priority).
            Only includes assigned targets.
        """
        coverage = {}
        
        for idx, (sat_id, tgt_id) in enumerate(self.feasible_pairs):
            if binary_vector[idx] == 1:
                priority = self.problem.get_target_priority(tgt_id)
                coverage[tgt_id] = (sat_id, priority)
        
        return coverage
    
    def solve(self) -> Dict:
        """
        Solve the problem using brute-force enumeration.
        
        Returns:
            Dict with keys:
                - 'optimal_vector': best binary vector found
                - 'objective_value': objective value at optimum
                - 'variable_names': list of variable names
                - 'assignments': dict of (sat_id, tgt_id) -> value for assigned pairs
                - 'feasible': boolean, whether optimal is feasible
                - 'num_feasible': count of feasible solutions evaluated
                - 'num_infeasible': count of infeasible solutions rejected
                - 'target_coverage': dict of assigned targets and priorities
                - 'total_priority_covered': sum of priorities of assigned targets
                - 'satellite_energy_usage': dict of sat_id -> energy usage
                - 'satellite_capacity_usage': dict of sat_id -> (num_targets, capacity)
                - 'constraint_info': detailed constraint status at optimum
        """
        best_vector = None
        best_obj = -np.inf
        best_feasible = False
        best_constraint_info = None
        
        # Enumerate all 2^num_variables binary combinations
        for binary_tuple in itertools.product([0, 1], repeat=self.num_variables):
            binary_vector = list(binary_tuple)
            
            # Check feasibility
            is_feasible, constraint_info = self._is_feasible(binary_vector)
            
            if is_feasible:
                self.num_feasible += 1
                obj_value = self._calculate_objective(binary_vector)
                
                # Store for debugging
                self.all_evaluations.append({
                    'vector': binary_vector,
                    'objective': obj_value,
                    'feasible': True
                })
                
                # Update best if this is better
                if obj_value > best_obj:
                    best_obj = obj_value
                    best_vector = binary_vector
                    best_feasible = True
                    best_constraint_info = constraint_info
            else:
                self.num_infeasible += 1
                self.all_evaluations.append({
                    'vector': binary_vector,
                    'objective': None,
                    'feasible': False,
                    'constraint_info': constraint_info
                })
        
        self.optimal_vector = best_vector
        self.optimal_value = best_obj
        
        # Build result dictionary
        result = {
            'optimal_vector': best_vector,
            'objective_value': best_obj if best_feasible else None,
            'variable_names': self.variable_names,
            'feasible': best_feasible,
            'num_feasible': self.num_feasible,
            'num_infeasible': self.num_infeasible,
        }
        
        if best_feasible:
            # Add detailed assignment information
            assignment = self._vector_to_assignment(best_vector)
            result['assignments'] = {pair: val for pair, val in assignment.items() if val == 1}
            
            # Add target coverage
            coverage = self._get_target_coverage(best_vector)
            result['target_coverage'] = coverage
            total_priority = sum(priority for _, priority in coverage.values())
            result['total_priority_covered'] = total_priority
            
            # Add energy and capacity usage
            result['satellite_energy_usage'] = best_constraint_info.get('satellite_energy', {})
            result['satellite_capacity_usage'] = {}
            for sat_id in self.problem.get_all_satellite_ids():
                num_assigned = best_constraint_info.get('satellite_counts', {}).get(sat_id, 0)
                capacity = self.problem.get_satellite_capacity(sat_id)
                result['satellite_capacity_usage'][sat_id] = (num_assigned, capacity)
            
            result['constraint_info'] = best_constraint_info
        
        return result
    
    def get_summary(self, result: Dict) -> str:
        """
        Get a human-readable summary of the optimization result.
        
        Args:
            result: Result dictionary from solve().
            
        Returns:
            Formatted summary string.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("CLASSICAL OPTIMIZER RESULT")
        lines.append("=" * 70)
        
        lines.append(f"\nProblem: {self.problem.get_problem_name()}")
        lines.append(f"Satellites: {self.problem.get_num_satellites()}")
        lines.append(f"Targets: {self.problem.get_num_targets()}")
        lines.append(f"Feasible variables: {self.num_variables}")
        
        lines.append(f"\nSolution feasibility: {'FEASIBLE' if result['feasible'] else 'INFEASIBLE'}")
        lines.append(f"Objective value: {result['objective_value']}")
        lines.append(f"Optimal vector: {result['optimal_vector']}")
        lines.append(f"Variable names: {result['variable_names']}")
        
        if result['feasible']:
            lines.append(f"\nAssignments:")
            for (sat_id, tgt_id), val in result['assignments'].items():
                if val == 1:
                    sat_name = self.problem.get_satellite(sat_id)['name']
                    tgt_name = self.problem.get_target(tgt_id)['name']
                    lines.append(f"  {sat_name} → {tgt_name}")
            
            lines.append(f"\nTarget coverage:")
            for tgt_id, (sat_id, priority) in result['target_coverage'].items():
                sat_name = self.problem.get_satellite(sat_id)['name']
                tgt_name = self.problem.get_target(tgt_id)['name']
                lines.append(f"  {tgt_name} (priority={priority}) ← {sat_name}")
            lines.append(f"  Total priority covered: {result['total_priority_covered']}")
            
            lines.append(f"\nEnergy usage:")
            for sat_id, energy in result['satellite_energy_usage'].items():
                budget = self.problem.get_satellite_energy_budget(sat_id)
                sat_name = self.problem.get_satellite(sat_id)['name']
                lines.append(f"  {sat_name}: {energy:.1f} / {budget} units")
            
            lines.append(f"\nCapacity usage:")
            for sat_id, (num_assigned, capacity) in result['satellite_capacity_usage'].items():
                sat_name = self.problem.get_satellite(sat_id)['name']
                lines.append(f"  {sat_name}: {num_assigned} / {capacity} targets")
        
        lines.append(f"\nSearch statistics:")
        lines.append(f"  Feasible solutions: {result['num_feasible']}")
        lines.append(f"  Infeasible solutions: {result['num_infeasible']}")
        lines.append(f"  Total combinations evaluated: {result['num_feasible'] + result['num_infeasible']}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
