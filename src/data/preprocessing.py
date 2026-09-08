"""
Data preprocessing module for Quantum Satellite Optimizer.

Loads, validates, and prepares satellite-target assignment problem instances
from JSON format. Identifies feasible assignment pairs based on sensor compatibility.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ProblemInstance:
    """
    Represents a satellite-target assignment problem instance.
    
    Loads problem data from JSON, validates structure, identifies feasible
    satellite-target pairs, and provides accessor methods for optimization.
    """
    
    def __init__(self, problem_path: str):
        """
        Initialize problem instance by loading and validating JSON file.
        
        Args:
            problem_path: Path to JSON problem file.
            
        Raises:
            FileNotFoundError: If problem file does not exist.
            ValueError: If problem structure is invalid.
        """
        self.problem_path = Path(problem_path)
        
        if not self.problem_path.exists():
            raise FileNotFoundError(f"Problem file not found: {problem_path}")
        
        with open(self.problem_path, 'r') as f:
            self.problem_data = json.load(f)
        
        self._validate()
        self._build_indices()
        self._identify_feasible_pairs()
    
    def _validate(self) -> None:
        """
        Validate problem structure and required fields.
        
        Raises:
            ValueError: If required fields are missing or invalid.
        """
        required_keys = ['satellites', 'targets', 'observation_costs', 'metadata']
        for key in required_keys:
            if key not in self.problem_data:
                raise ValueError(f"Missing required key in problem file: {key}")
        
        # Validate satellites
        if not isinstance(self.problem_data['satellites'], list):
            raise ValueError("'satellites' must be a list")
        if len(self.problem_data['satellites']) == 0:
            raise ValueError("At least one satellite is required")
        
        for sat in self.problem_data['satellites']:
            required_sat_keys = ['id', 'name', 'type', 'capacity', 'energy_budget']
            for key in required_sat_keys:
                if key not in sat:
                    raise ValueError(f"Satellite missing required field: {key}")
        
        # Validate targets
        if not isinstance(self.problem_data['targets'], list):
            raise ValueError("'targets' must be a list")
        if len(self.problem_data['targets']) == 0:
            raise ValueError("At least one target is required")
        
        for tgt in self.problem_data['targets']:
            required_tgt_keys = ['id', 'name', 'priority', 'compatible_sensors']
            for key in required_tgt_keys:
                if key not in tgt:
                    raise ValueError(f"Target missing required field: {key}")
        
        # Validate observation costs
        if not isinstance(self.problem_data['observation_costs'], list):
            raise ValueError("'observation_costs' must be a list")
        
        for cost_entry in self.problem_data['observation_costs']:
            required_cost_keys = ['satellite_id', 'target_id', 'cost']
            for key in required_cost_keys:
                if key not in cost_entry:
                    raise ValueError(f"Observation cost entry missing required field: {key}")
        
        # Validate metadata
        metadata = self.problem_data['metadata']
        required_meta_keys = ['num_satellites', 'num_targets', 'objective_weights', 'penalty_weights']
        for key in required_meta_keys:
            if key not in metadata:
                raise ValueError(f"Metadata missing required field: {key}")
    
    def _build_indices(self) -> None:
        """Build lookup dictionaries for fast access to satellites and targets."""
        self.satellites_by_id = {sat['id']: sat for sat in self.problem_data['satellites']}
        self.targets_by_id = {tgt['id']: tgt for tgt in self.problem_data['targets']}
        
        # Build cost matrix as dict for fast lookup
        self.cost_matrix = {}
        for cost_entry in self.problem_data['observation_costs']:
            sat_id = cost_entry['satellite_id']
            tgt_id = cost_entry['target_id']
            cost = cost_entry['cost']
            self.cost_matrix[(sat_id, tgt_id)] = cost
    
    def _identify_feasible_pairs(self) -> None:
        """
        Identify feasible satellite-target pairs based on sensor compatibility.
        
        A pair (satellite_id, target_id) is feasible if and only if:
        1. The satellite's sensor type is in the target's compatible_sensors list.
        2. An observation cost entry exists for this pair.
        
        Both conditions must be true for a pair to be feasible.
        """
        self.feasible_pairs = []  # List of (sat_id, tgt_id) tuples
        
        for sat in self.problem_data['satellites']:
            sat_id = sat['id']
            sat_type = sat['type']
            
            for tgt in self.problem_data['targets']:
                tgt_id = tgt['id']
                compatible_sensors = tgt['compatible_sensors']
                
                # Check both conditions:
                # 1. Sensor type compatibility
                # 2. Cost exists for this pair
                if sat_type in compatible_sensors and (sat_id, tgt_id) in self.cost_matrix:
                    self.feasible_pairs.append((sat_id, tgt_id))
        
        # Sort for consistent ordering
        self.feasible_pairs.sort()
    
    def get_feasible_pairs(self) -> List[Tuple[int, int]]:
        """
        Get list of feasible (satellite_id, target_id) pairs.
        
        Returns:
            List of (sat_id, tgt_id) tuples in sorted order.
        """
        return self.feasible_pairs
    
    def get_variable_names(self) -> List[str]:
        """
        Get variable names in standard order x_ij.
        
        Returns:
            List of variable names like ['x11', 'x12', 'x21', 'x23'].
        """
        names = []
        for sat_id, tgt_id in self.feasible_pairs:
            names.append(f"x{sat_id}{tgt_id}")
        return names
    
    def get_num_satellites(self) -> int:
        """Get number of satellites."""
        return len(self.problem_data['satellites'])
    
    def get_num_targets(self) -> int:
        """Get number of targets."""
        return len(self.problem_data['targets'])
    
    def get_num_variables(self) -> int:
        """Get number of decision variables (feasible pairs)."""
        return len(self.feasible_pairs)
    
    def get_satellite(self, sat_id: int) -> Dict:
        """
        Get satellite information by ID.
        
        Args:
            sat_id: Satellite ID.
            
        Returns:
            Satellite dictionary with keys: id, name, type, capacity, energy_budget.
            
        Raises:
            KeyError: If satellite ID not found.
        """
        if sat_id not in self.satellites_by_id:
            raise KeyError(f"Satellite ID {sat_id} not found")
        return self.satellites_by_id[sat_id]
    
    def get_target(self, tgt_id: int) -> Dict:
        """
        Get target information by ID.
        
        Args:
            tgt_id: Target ID.
            
        Returns:
            Target dictionary with keys: id, name, priority, compatible_sensors.
            
        Raises:
            KeyError: If target ID not found.
        """
        if tgt_id not in self.targets_by_id:
            raise KeyError(f"Target ID {tgt_id} not found")
        return self.targets_by_id[tgt_id]
    
    def get_target_priority(self, tgt_id: int) -> float:
        """
        Get priority value of a target.
        
        Args:
            tgt_id: Target ID.
            
        Returns:
            Priority value (numeric).
            
        Raises:
            KeyError: If target ID not found.
        """
        target = self.get_target(tgt_id)
        return target['priority']
    
    def get_observation_cost(self, sat_id: int, tgt_id: int) -> Optional[float]:
        """
        Get observation cost for a satellite-target pair.
        
        Args:
            sat_id: Satellite ID.
            tgt_id: Target ID.
            
        Returns:
            Cost value if pair is feasible, None if not feasible or not found.
        """
        return self.cost_matrix.get((sat_id, tgt_id), None)
    
    def is_compatible(self, sat_id: int, tgt_id: int) -> bool:
        """
        Check if a satellite can observe a target (sensor compatibility and cost exists).
        
        A pair is compatible if and only if:
        1. The satellite's sensor type is in the target's compatible_sensors list.
        2. An observation cost exists for this pair.
        
        Args:
            sat_id: Satellite ID.
            tgt_id: Target ID.
            
        Returns:
            True if both conditions are met, False otherwise.
        """
        # Condition 1: Cost must exist
        if (sat_id, tgt_id) not in self.cost_matrix:
            return False
        
        # Condition 2: Sensor type must be compatible
        try:
            satellite = self.get_satellite(sat_id)
            target = self.get_target(tgt_id)
            sat_type = satellite['type']
            compatible_sensors = target['compatible_sensors']
            return sat_type in compatible_sensors
        except KeyError:
            return False
    
    def get_satellite_capacity(self, sat_id: int) -> int:
        """
        Get capacity (max targets) of a satellite.
        
        Args:
            sat_id: Satellite ID.
            
        Returns:
            Capacity value.
            
        Raises:
            KeyError: If satellite ID not found.
        """
        return self.get_satellite(sat_id)['capacity']
    
    def get_satellite_energy_budget(self, sat_id: int) -> float:
        """
        Get energy budget of a satellite.
        
        Args:
            sat_id: Satellite ID.
            
        Returns:
            Energy budget value.
            
        Raises:
            KeyError: If satellite ID not found.
        """
        return self.get_satellite(sat_id)['energy_budget']
    
    def get_objective_weights(self) -> Dict[str, float]:
        """
        Get objective weights (w_priority, w_cost).
        
        Returns:
            Dict with keys 'w_priority' and 'w_cost'.
        """
        return self.problem_data['metadata']['objective_weights']
    
    def get_penalty_weights(self) -> Dict[str, float]:
        """
        Get penalty weights for QUBO constraints.
        
        Returns:
            Dict with keys 'lambda1', 'lambda2', etc.
        """
        return self.problem_data['metadata']['penalty_weights']
    
    def get_problem_name(self) -> str:
        """Get name of the problem instance."""
        return self.problem_data.get('problem_name', 'Unnamed Problem')
    
    def get_problem_description(self) -> str:
        """Get description of the problem instance."""
        return self.problem_data.get('description', '')
    
    def get_all_satellite_ids(self) -> List[int]:
        """Get list of all satellite IDs."""
        return sorted([sat['id'] for sat in self.problem_data['satellites']])
    
    def get_all_target_ids(self) -> List[int]:
        """Get list of all target IDs."""
        return sorted([tgt['id'] for tgt in self.problem_data['targets']])
    
    def get_all_observation_costs(self) -> Dict[Tuple[int, int], float]:
        """
        Get all observation costs as a dictionary.
        
        Returns:
            Dict mapping (sat_id, tgt_id) -> cost for all feasible pairs.
        """
        return dict(self.cost_matrix)


def load_problem(problem_path: str) -> ProblemInstance:
    """
    Load a problem instance from JSON file.
    
    Args:
        problem_path: Path to JSON problem file.
        
    Returns:
        ProblemInstance object.
        
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If structure is invalid.
    """
    return ProblemInstance(problem_path)
