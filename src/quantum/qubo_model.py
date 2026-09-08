"""
QUBO Model: Quantum Unconstrained Binary Optimization Formulation

This module converts the satellite-target assignment problem into QUBO form
suitable for QAOA and other quantum annealers.

Mathematical Foundation:
=======================

ORIGINAL PROBLEM (Integer Linear Program):
-------------------------------------------
Variables: x[i,j] ∈ {0,1}  (i=satellite, j=target)

Objective (MAXIMIZE):
  Z = Σ(i,j) c[i,j] * x[i,j]
  where c[i,j] = target_priority[j] * feasibility[i,j]

Constraints:
  C1 (Fuel): Σ(j) fuel_cost[i,j] * x[i,j] ≤ available_fuel[i]  ∀i
  C2 (Capacity): Σ(j) x[i,j] ≤ observation_capacity[i]  ∀i
  C3 (Exclusivity): Σ(i) x[i,j] ≤ 1  ∀j

QUBO CONVERSION:
================

Since QUBO solvers MINIMIZE unconstrained binary problems, we reformulate:

1. OBJECTIVE FLIP (minimize negative objective):
   Minimize: -Z = -Σ(i,j) c[i,j] * x[i,j]
   Or equivalently, use penalty method with large penalty weights.

2. CONSTRAINT PENALTIES:
   
   C1 Penalty (Fuel per satellite):
   For each satellite i, if Σ(j) fuel_cost[i,j] * x[i,j] > available_fuel[i],
   penalize violation:
   
   P_C1 = λ_fuel * Σ(i) (Σ(j) fuel_cost[i,j] * x[i,j] - available_fuel[i])²
   
   Expanding (a - b)² = a² - 2ab + b² where a = Σ fuel_cost*x, b = budget:
   
   P_C1 = λ_fuel * Σ(i) [
     (Σ(j) fuel_cost[i,j] * x[i,j])² - 2*available_fuel[i]*Σ(j) fuel_cost[i,j]*x[i,j] + available_fuel[i]²
   ]
   
   The (Σ fuel*x)² term expands to quadratic form when multiplied out.
   
   C2 Penalty (Capacity per satellite):
   P_C2 = λ_cap * Σ(i) (Σ(j) x[i,j] - observation_capacity[i])²
   
   Expanding similarly:
   P_C2 = λ_cap * Σ(i) [
     (Σ(j) x[i,j])² - 2*observation_capacity[i]*Σ(j) x[i,j] + observation_capacity[i]²
   ]
   
   Note: (Σ x[i,j])² = Σ(j) x[i,j]² + 2*Σ(j<k) x[i,j]*x[i,k]
   Since x[i,j] ∈ {0,1}, x[i,j]² = x[i,j], so:
   (Σ x[i,j])² = Σ(j) x[i,j] + 2*Σ(j<k) x[i,j]*x[i,k]
   
   C3 Penalty (Exclusivity per target):
   P_C3 = λ_excl * Σ(j) (Σ(i) x[i,j] - 1)²
   
   Expanding:
   P_C3 = λ_excl * Σ(j) [
     (Σ(i) x[i,j])² - 2*Σ(i) x[i,j] + 1
   ]

3. COMPLETE QUBO OBJECTIVE:
   E = -Z + P_C1 + P_C2 + P_C3
   
   This is rewritten in canonical QUBO form:
   E = Σ(i,j) Q[i,j] * x[i] * x[j]  (where i ≤ j in matrix indexing)

NUMERICAL EXAMPLE (2 Satellites × 3 Targets):
==============================================

Data:
SAT_001: fuel_avail=100, capacity=2
SAT_002: fuel_avail=80, capacity=1

TGT_001: priority=9
TGT_002: priority=7
TGT_003: priority=5

Cost matrix (fuel_cost, feasibility):
           TGT_001       TGT_002       TGT_003
SAT_001: (25, 0.98)  (30, 0.85)  (35, 0.90)
SAT_002: (40, 0.80)  (20, 0.95)  (45, 0.75)

Objective coefficients c[i,j] = priority[j] * feasibility[i,j]:
           TGT_001       TGT_002       TGT_003
SAT_001: 9*0.98=8.82  7*0.85=5.95  5*0.90=4.50
SAT_002: 9*0.80=7.20  7*0.95=6.65  5*0.75=3.75

Variable mapping (linear indexing):
  x0 = x[SAT_001, TGT_001]
  x1 = x[SAT_001, TGT_002]
  x2 = x[SAT_001, TGT_003]
  x3 = x[SAT_002, TGT_001]
  x4 = x[SAT_002, TGT_002]
  x5 = x[SAT_002, TGT_003]

Penalty weights (typically large to ensure constraint satisfaction):
  λ_fuel = 100
  λ_cap = 100
  λ_excl = 100

The resulting Q matrix is 6×6 with:
  - Diagonal entries: linear coefficients (from objective flip and constraint penalties)
  - Off-diagonal entries: quadratic interaction terms (from constraint penalty expansions)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class QuboMapping:
    """Mapping between QUBO variables and satellite-target assignments."""
    var_index_to_assignment: Dict[int, Tuple[str, str]]  # var_idx -> (sat_id, tgt_id)
    assignment_to_var_index: Dict[Tuple[str, str], int]  # (sat_id, tgt_id) -> var_idx
    n_variables: int


class QuboModel:
    """
    Convert satellite-target assignment problem to QUBO form.
    
    The QUBO (Quadratic Unconstrained Binary Optimization) formulation:
      E(x) = Σ(i≤j) Q[i,j] * x[i] * x[j]
    
    where x[i] ∈ {0,1} are binary decision variables and Q is the QUBO matrix.
    """
    
    def __init__(
        self,
        satellites_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        cost_df: pd.DataFrame,
        penalty_weight_fuel: float = 100.0,
        penalty_weight_capacity: float = 100.0,
        penalty_weight_exclusivity: float = 100.0
    ):
        """
        Initialize QUBO model.
        
        Parameters:
        -----------
        satellites_df, targets_df, cost_df : pd.DataFrame
            Input data (same format as classical optimizer)
        penalty_weight_* : float
            Lagrange multipliers for constraint penalties.
            Higher values enforce constraints more strictly.
        """
        self.satellites_df = satellites_df.copy()
        self.targets_df = targets_df.copy()
        self.cost_df = cost_df.copy()
        
        self.satellite_ids = satellites_df['satellite_id'].tolist()
        self.target_ids = targets_df['target_id'].tolist()
        
        self.n_satellites = len(self.satellite_ids)
        self.n_targets = len(self.target_ids)
        self.n_variables = self.n_satellites * self.n_targets
        
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
        
        # Build cost matrix: (fuel_cost, feasibility)
        self.cost_matrix = {}
        for _, row in cost_df.iterrows():
            sat_id = row['satellite_id']
            tgt_id = row['target_id']
            self.cost_matrix[(sat_id, tgt_id)] = {
                'fuel_cost': row['fuel_cost'],
                'feasibility': row['feasibility']
            }
        
        # Penalty weights
        self.λ_fuel = penalty_weight_fuel
        self.λ_cap = penalty_weight_capacity
        self.λ_excl = penalty_weight_exclusivity
        
        # Create variable mapping
        self.mapping = self._create_mapping()
        
        # Build Q matrix
        self.Q = self._build_Q_matrix()
    
    def _create_mapping(self) -> QuboMapping:
        """
        Create bijection between variable indices and satellite-target assignments.
        
        Variables indexed by: sat_idx * n_targets + tgt_idx
        """
        var_index_to_assignment = {}
        assignment_to_var_index = {}
        
        var_idx = 0
        for sat_idx, sat_id in enumerate(self.satellite_ids):
            for tgt_idx, tgt_id in enumerate(self.target_ids):
                assignment = (sat_id, tgt_id)
                var_index_to_assignment[var_idx] = assignment
                assignment_to_var_index[assignment] = var_idx
                var_idx += 1
        
        return QuboMapping(
            var_index_to_assignment=var_index_to_assignment,
            assignment_to_var_index=assignment_to_var_index,
            n_variables=self.n_variables
        )
    
    def _build_Q_matrix(self) -> np.ndarray:
        """
        Build Q matrix for QUBO formulation.
        
        E = -Σ(i,j) c[i,j] * x[i,j] + P_C1 + P_C2 + P_C3
        
        Returns:
        --------
        Q : np.ndarray, shape (n_variables, n_variables)
            Symmetric QUBO matrix. Q[i,j] is the coefficient of x[i]*x[j].
        """
        Q = np.zeros((self.n_variables, self.n_variables))
        
        # 1. OBJECTIVE TERM: -Σ(i,j) c[i,j] * x[i,j]
        # This contributes to diagonal (x[i]*x[i] = x[i] for binary variables)
        for var_idx in range(self.n_variables):
            sat_id, tgt_id = self.mapping.var_index_to_assignment[var_idx]
            c_ij = (
                self.target_priority[tgt_id] *
                self.cost_matrix[(sat_id, tgt_id)]['feasibility']
            )
            Q[var_idx, var_idx] -= c_ij  # Negative because we minimize
        
        # 2. FUEL CONSTRAINT PENALTY: λ_fuel * Σ(i) (Σ(j) fuel*x - budget)²
        self._add_fuel_penalty(Q)
        
        # 3. CAPACITY CONSTRAINT PENALTY: λ_cap * Σ(i) (Σ(j) x - capacity)²
        self._add_capacity_penalty(Q)
        
        # 4. EXCLUSIVITY CONSTRAINT PENALTY: λ_excl * Σ(j) (Σ(i) x - 1)²
        self._add_exclusivity_penalty(Q)
        
        return Q
    
    def _add_fuel_penalty(self, Q: np.ndarray) -> None:
        """
        Add fuel constraint penalty term to Q matrix.
        
        P_C1 = λ_fuel * Σ(i) (Σ(j) fuel[i,j]*x[i,j] - budget[i])²
        
        Expanding (a - b)² = a² - 2ab + b²:
        - a² = (Σ fuel*x)² = Σ(j,k) fuel[i,j]*fuel[i,k]*x[i,j]*x[i,k]
        - 2ab = 2*budget*Σ fuel*x
        - b² = constant (ignored in QUBO as it doesn't affect optimization)
        """
        for sat_idx, sat_id in enumerate(self.satellite_ids):
            budget = self.satellite_fuel[sat_id]
            
            # Get all variables corresponding to this satellite
            sat_vars = [
                sat_idx * self.n_targets + tgt_idx
                for tgt_idx in range(self.n_targets)
            ]
            
            # Quadratic term: (Σ fuel*x)² 
            # Expands to Σ(j,k) fuel[j]*fuel[k]*x[j]*x[k]
            for var_j in sat_vars:
                _, tgt_j = self.mapping.var_index_to_assignment[var_j]
                fuel_j = self.cost_matrix[(sat_id, tgt_j)]['fuel_cost']
                
                for var_k in sat_vars:
                    _, tgt_k = self.mapping.var_index_to_assignment[var_k]
                    fuel_k = self.cost_matrix[(sat_id, tgt_k)]['fuel_cost']
                    
                    coeff = self.λ_fuel * fuel_j * fuel_k
                    if var_j < var_k:
                        Q[var_j, var_k] += 2 * coeff  # Factor of 2 for off-diagonal
                    elif var_j == var_k:
                        Q[var_j, var_j] += coeff
            
            # Linear term: -2*budget*Σ fuel*x
            for var_j in sat_vars:
                _, tgt_j = self.mapping.var_index_to_assignment[var_j]
                fuel_j = self.cost_matrix[(sat_id, tgt_j)]['fuel_cost']
                Q[var_j, var_j] -= 2 * self.λ_fuel * budget * fuel_j
    
    def _add_capacity_penalty(self, Q: np.ndarray) -> None:
        """
        Add capacity constraint penalty term to Q matrix.
        
        P_C2 = λ_cap * Σ(i) (Σ(j) x[i,j] - capacity[i])²
        
        Note: x² = x for binary variables, so:
        (Σ x)² = Σ x + 2*Σ(j<k) x[j]*x[k]
        """
        for sat_idx, sat_id in enumerate(self.satellite_ids):
            capacity = self.satellite_capacity[sat_id]
            
            # Get all variables corresponding to this satellite
            sat_vars = [
                sat_idx * self.n_targets + tgt_idx
                for tgt_idx in range(self.n_targets)
            ]
            
            # Quadratic term: (Σ x)² = Σ x + 2*Σ(j<k) x[j]*x[k]
            # Diagonal part: x[j]*x[j] = x[j]
            for var_j in sat_vars:
                Q[var_j, var_j] += self.λ_cap
                
                # Off-diagonal: 2*x[j]*x[k]
                for var_k in sat_vars:
                    if var_j < var_k:
                        Q[var_j, var_k] += 2 * self.λ_cap
            
            # Linear term: -2*capacity*Σ x
            for var_j in sat_vars:
                Q[var_j, var_j] -= 2 * self.λ_cap * capacity
    
    def _add_exclusivity_penalty(self, Q: np.ndarray) -> None:
        """
        Add exclusivity constraint penalty term to Q matrix.
        
        P_C3 = λ_excl * Σ(j) (Σ(i) x[i,j] - 1)²
        
        For each target j, penalty is on (Σ x[i,j] - 1)²
        """
        for tgt_idx, tgt_id in enumerate(self.target_ids):
            # Get all variables corresponding to this target
            tgt_vars = [
                sat_idx * self.n_targets + tgt_idx
                for sat_idx in range(self.n_satellites)
            ]
            
            # Quadratic term: (Σ x)² = Σ x + 2*Σ(i<k) x[i]*x[k]
            for var_i in tgt_vars:
                Q[var_i, var_i] += self.λ_excl
                
                for var_k in tgt_vars:
                    if var_i < var_k:
                        Q[var_i, var_k] += 2 * self.λ_excl
            
            # Linear term: -2*1*Σ x = -2*Σ x
            for var_i in tgt_vars:
                Q[var_i, var_i] -= 2 * self.λ_excl
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate QUBO objective for a given assignment vector.
        
        E(x) = Σ(i≤j) Q[i,j] * x[i] * x[j]
        
        Parameters:
        -----------
        x : np.ndarray
            Binary vector of length n_variables
        
        Returns:
        --------
        float : QUBO objective value
        """
        return float(x @ self.Q @ x)
    
    def decode_solution(self, x: np.ndarray) -> Dict[str, any]:
        """
        Decode QUBO solution vector into human-readable format.
        
        Parameters:
        -----------
        x : np.ndarray
            Binary vector of length n_variables
        
        Returns:
        --------
        Dict with keys:
          - 'assignments': List[(sat_id, tgt_id), ...]
          - 'qubo_objective': QUBO objective value
          - 'objective_value': Negative QUBO objective (approximates original objective)
          - 'fuel_used': {sat_id: fuel_used}
          - 'targets_observed': {tgt_id: sat_id or None}
        """
        assignments = []
        fuel_used = {sat_id: 0.0 for sat_id in self.satellite_ids}
        targets_observed = {tgt_id: None for tgt_id in self.target_ids}
        objective_value = 0.0
        
        for var_idx in range(self.n_variables):
            if x[var_idx] > 0.5:  # Treat as 1 (allows for approximate solutions)
                sat_id, tgt_id = self.mapping.var_index_to_assignment[var_idx]
                assignments.append((sat_id, tgt_id))
                
                fuel_cost = self.cost_matrix[(sat_id, tgt_id)]['fuel_cost']
                fuel_used[sat_id] += fuel_cost
                targets_observed[tgt_id] = sat_id
                
                c_ij = (
                    self.target_priority[tgt_id] *
                    self.cost_matrix[(sat_id, tgt_id)]['feasibility']
                )
                objective_value += c_ij
        
        qubo_objective = self.evaluate(x)
        
        return {
            'assignments': assignments,
            'qubo_objective': qubo_objective,
            'objective_value': objective_value,
            'fuel_used': fuel_used,
            'targets_observed': targets_observed
        }
