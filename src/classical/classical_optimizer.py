"""
Classical Satellite-Target Assignment Optimizer

This module formulates and solves the satellite-target assignment problem
using integer linear programming (ILP) via SciPy's optimization tools
and OR-Tools as a fallback.

Problem formulation:
- Decision variables: x[i,j] binary (satellite i observes target j)
- Objective: Maximize total observation value (priority × feasibility)
- Constraints:
  1. Fuel budget per satellite
  2. Observation capacity per satellite
  3. Each target observed by at most one satellite (optional)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ClassicalSolution:
    """Result of classical optimization."""
    assignments: List[Tuple[str, str]]  # [(satellite_id, target_id), ...]
    objective_value: float
    total_fuel_used: Dict[str, float]  # {satellite_id: fuel_used}
    target_coverage: Dict[str, str]  # {target_id: satellite_id or None}
    constraint_violations: List[str]  # List of constraint violations (empty if feasible)
    is_feasible: bool


class ClassicalOptimizer:
    """
    Classical satellite-target assignment optimizer using ILP.
    
    Solves the satellite constellation optimization problem:
    - Maximize observation value
    - Subject to fuel budget and capacity constraints
    """
    
    def __init__(
        self,
        satellites_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        cost_df: pd.DataFrame
    ):
        """
        Initialize optimizer with satellite, target, and cost data.
        
        Parameters:
        -----------
        satellites_df : pd.DataFrame
            Columns: satellite_id, available_fuel, observation_capacity
        targets_df : pd.DataFrame
            Columns: target_id, target_priority, observation_feasibility
        cost_df : pd.DataFrame
            Columns: satellite_id, target_id, fuel_cost, feasibility
        """
        self.satellites_df = satellites_df.copy()
        self.targets_df = targets_df.copy()
        self.cost_df = cost_df.copy()
        
        self.satellite_ids = satellites_df['satellite_id'].tolist()
        self.target_ids = targets_df['target_id'].tolist()
        
        # Build lookup dictionaries
        self.satellite_fuel = dict(
            zip(satellites_df['satellite_id'], satellites_df['available_fuel'])
        )
        self.satellite_capacity = dict(
            zip(satellites_df['satellite_id'], satellites_df['observation_capacity'])
        )
        self.target_priority = dict(
            zip(targets_df['target_id'], targets_df['target_priority'])
        )
        self.target_feasibility = dict(
            zip(targets_df['target_id'], targets_df['observation_feasibility'])
        )
        
        # Build cost matrix: cost_matrix[i][j] = (fuel_cost, feasibility) for sat[i] -> tgt[j]
        self.cost_matrix = {}
        for _, row in cost_df.iterrows():
            sat_id = row['satellite_id']
            tgt_id = row['target_id']
            self.cost_matrix[(sat_id, tgt_id)] = {
                'fuel_cost': row['fuel_cost'],
                'feasibility': row['feasibility']
            }
    
    def _enumerate_feasible_solutions(self) -> List[Dict[str, Any]]:
        """
        Enumerate all feasible solutions by brute-force search.
        
        For small problem sizes (2 satellites, 3 targets = 2^6 = 64 possible assignments),
        exhaustive enumeration is practical and guarantees global optimum.
        
        Returns:
        --------
        List[Dict] : List of feasible solutions, each containing:
            - 'assignments': list of (sat_id, tgt_id) tuples
            - 'objective_value': total observation value
            - 'fuel_used': dict of {sat_id: fuel_used}
            - 'target_coverage': dict of {tgt_id: sat_id or None}
        """
        n_satellites = len(self.satellite_ids)
        n_targets = len(self.target_ids)
        n_vars = n_satellites * n_targets
        
        feasible_solutions = []
        
        # Iterate over all 2^n possible assignments
        for assignment_bits in range(2 ** n_vars):
            # Decode bits into assignment matrix x[i,j]
            x = {}
            bit_index = 0
            for sat_idx, sat_id in enumerate(self.satellite_ids):
                for tgt_idx, tgt_id in enumerate(self.target_ids):
                    x[(sat_id, tgt_id)] = (assignment_bits >> bit_index) & 1
                    bit_index += 1
            
            # Check feasibility of this assignment
            violations = []
            fuel_used = {sat_id: 0.0 for sat_id in self.satellite_ids}
            target_coverage = {tgt_id: None for tgt_id in self.target_ids}
            
            # Constraint 1: Fuel budget per satellite
            for sat_id in self.satellite_ids:
                fuel = 0.0
                for tgt_id in self.target_ids:
                    if x[(sat_id, tgt_id)] == 1:
                        fuel += self.cost_matrix[(sat_id, tgt_id)]['fuel_cost']
                fuel_used[sat_id] = fuel
                
                if fuel > self.satellite_fuel[sat_id]:
                    violations.append(
                        f"Satellite {sat_id} fuel exceeded: {fuel} > {self.satellite_fuel[sat_id]}"
                    )
            
            # Constraint 2: Observation capacity per satellite
            for sat_id in self.satellite_ids:
                n_targets_observed = sum(
                    x[(sat_id, tgt_id)] for tgt_id in self.target_ids
                )
                if n_targets_observed > self.satellite_capacity[sat_id]:
                    violations.append(
                        f"Satellite {sat_id} capacity exceeded: {n_targets_observed} > {self.satellite_capacity[sat_id]}"
                    )
            
            # Constraint 3: Each target observed by at most one satellite
            for tgt_id in self.target_ids:
                n_satellites_observing = sum(
                    x[(sat_id, tgt_id)] for sat_id in self.satellite_ids
                )
                if n_satellites_observing > 1:
                    violations.append(
                        f"Target {tgt_id} observed by multiple satellites: {n_satellites_observing}"
                    )
                elif n_satellites_observing == 1:
                    # Find which satellite
                    for sat_id in self.satellite_ids:
                        if x[(sat_id, tgt_id)] == 1:
                            target_coverage[tgt_id] = sat_id
            
            # If feasible, compute objective value
            if not violations:
                objective_value = 0.0
                assignments = []
                for sat_id in self.satellite_ids:
                    for tgt_id in self.target_ids:
                        if x[(sat_id, tgt_id)] == 1:
                            # Objective: priority × feasibility
                            value = (
                                self.target_priority[tgt_id] *
                                self.cost_matrix[(sat_id, tgt_id)]['feasibility']
                            )
                            objective_value += value
                            assignments.append((sat_id, tgt_id))
                
                feasible_solutions.append({
                    'assignments': assignments,
                    'objective_value': objective_value,
                    'fuel_used': fuel_used.copy(),
                    'target_coverage': target_coverage.copy()
                })
        
        return feasible_solutions
    
    def solve(self) -> ClassicalSolution:
        """
        Solve the satellite-target assignment problem.
        
        For the small 2×3 problem, uses exhaustive enumeration.
        Finds the feasible solution with maximum objective value.
        
        Returns:
        --------
        ClassicalSolution : Solution object with assignments, objective value, and metadata.
        """
        feasible_solutions = self._enumerate_feasible_solutions()
        
        if not feasible_solutions:
            # No feasible solution exists; return empty result
            return ClassicalSolution(
                assignments=[],
                objective_value=0.0,
                total_fuel_used={sat_id: 0.0 for sat_id in self.satellite_ids},
                target_coverage={tgt_id: None for tgt_id in self.target_ids},
                constraint_violations=["No feasible solution exists"],
                is_feasible=False
            )
        
        # Find solution with maximum objective value
        best_solution = max(feasible_solutions, key=lambda s: s['objective_value'])
        
        return ClassicalSolution(
            assignments=best_solution['assignments'],
            objective_value=best_solution['objective_value'],
            total_fuel_used=best_solution['fuel_used'],
            target_coverage=best_solution['target_coverage'],
            constraint_violations=[],
            is_feasible=True
        )


def load_and_solve(
    satellites_path: str,
    targets_path: str,
    cost_path: str
) -> ClassicalSolution:
    """
    Convenience function: load data files and solve.
    
    Parameters:
    -----------
    satellites_path, targets_path, cost_path : str
        Paths to CSV data files.
    
    Returns:
    --------
    ClassicalSolution : Optimization result.
    """
    satellites_df = pd.read_csv(satellites_path)
    targets_df = pd.read_csv(targets_path)
    cost_df = pd.read_csv(cost_path)
    
    optimizer = ClassicalOptimizer(satellites_df, targets_df, cost_df)
    return optimizer.solve()
