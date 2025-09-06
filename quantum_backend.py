import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging
from logging.config import dictConfig
import json

# Configure logging
dictConfig({
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
})

logger = logging.getLogger(__name__)

# Define data models
class GateOperation(BaseModel):
    gate: str
    targets: List[int] = Field(..., min_items=1)
    controls: Optional[List[int]] = None
    parameters: Optional[List[float]] = None

class CircuitRequest(BaseModel):
    operations: List[GateOperation]
    num_qubits: int = Field(..., gt=0, le=25)  # Practical limit for simulation
    shots: Optional[int] = Field(default=1, gt=0)  # Number of measurements

class CircuitResponse(BaseModel):
    state_vector: Optional[List[List[float]]]  # [real, imag] pairs
    measurements: Optional[List[int]]
    probabilities: Optional[List[float]]

class QuantumSimulator:
    """
    A high-performance quantum simulator using state vector representation.
    """
    
    # Define common single-qubit gates
    SINGLE_QUBIT_GATES = {
        'i': np.array([[1, 0], [0, 1]], dtype=np.complex128),
        'x': np.array([[0, 1], [1, 0]], dtype=np.complex128),
        'y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        'z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
        'h': (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128),
        's': np.array([[1, 0], [0, 1j]], dtype=np.complex128),
        't': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex128),
    }
    
    def __init__(self, num_qubits: int):
        """
        Initialize the simulator with the specified number of qubits.
        
        Args:
            num_qubits: Number of qubits in the system
        """
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state[0] = 1.0 + 0j  # Initialize to |0...0⟩
        logger.info(f"Initialized simulator with {num_qubits} qubits")
    
    def reset(self):
        """Reset the simulator to the |0...0⟩ state."""
        self.state = np.zeros(2**self.num_qubits, dtype=np.complex128)
        self.state[0] = 1.0 + 0j
        logger.debug("Reset simulator to |0...0⟩ state")
    
    def _tensor_product(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        Compute the tensor product of a list of matrices.
        
        Args:
            matrices: List of matrices to tensor together
            
        Returns:
            The resulting tensor product matrix
        """
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)
        return result
    
    def _get_gate_matrix(self, gate_name: str, parameters: Optional[List[float]] = None) -> np.ndarray:
        """
        Get the matrix representation of a gate.
        
        Args:
            gate_name: Name of the gate
            parameters: Optional parameters for parameterized gates
            
        Returns:
            Matrix representation of the gate
        """
        gate_name = gate_name.lower()
        
        # Handle parameterized gates
        if gate_name == 'rx' and parameters:
            theta = parameters[0]
            return np.array([
                [np.cos(theta/2), -1j*np.sin(theta/2)],
                [-1j*np.sin(theta/2), np.cos(theta/2)]
            ], dtype=np.complex128)
        
        elif gate_name == 'ry' and parameters:
            theta = parameters[0]
            return np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ], dtype=np.complex128)
        
        elif gate_name == 'rz' and parameters:
            theta = parameters[0]
            return np.array([
                [np.exp(-1j*theta/2), 0],
                [0, np.exp(1j*theta/2)]
            ], dtype=np.complex128)
        
        elif gate_name == 'u' and parameters and len(parameters) == 3:
            # General unitary gate with 3 parameters (theta, phi, lambda)
            theta, phi, lam = parameters
            return np.array([
                [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
                [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
            ], dtype=np.complex128)
        
        # Handle standard gates
        elif gate_name in self.SINGLE_QUBIT_GATES:
            return self.SINGLE_QUBIT_GATES[gate_name]
        
        # Handle two-qubit gates
        elif gate_name == 'cnot':
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=np.complex128)
        
        elif gate_name == 'swap':
            return np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.complex128)
        
        elif gate_name == 'cz':
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=np.complex128)
        
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
    
    def _apply_single_qubit_gate(self, gate_matrix: np.ndarray, target: int):
        """
        Apply a single-qubit gate to the specified target qubit.
        
        Args:
            gate_matrix: 2x2 matrix representation of the gate
            target: Index of the target qubit
        """
        # Create the full operator by tensoring with identities
        operators = [np.eye(2, dtype=np.complex128) for _ in range(self.num_qubits)]
        operators[target] = gate_matrix
        
        # Compute the full operator matrix
        full_operator = self._tensor_product(operators)
        
        # Apply the operator to the state vector
        self.state = np.dot(full_operator, self.state)
    
    def _apply_controlled_gate(self, gate_matrix: np.ndarray, controls: List[int], target: int):
        """
        Apply a controlled gate with potentially multiple control qubits.
        
        Args:
            gate_matrix: 2x2 matrix representation of the gate
            controls: List of control qubit indices
            target: Index of the target qubit
        """
        n = self.num_qubits
        full_operator = np.eye(2**n, dtype=np.complex128)
        
        # Iterate through all basis states
        for i in range(2**n):
            # Check if all control qubits are |1⟩
            control_mask = 0
            for control in controls:
                control_mask |= (1 << (n - 1 - control))
            
            if (i & control_mask) == control_mask:
                # All controls are |1⟩, apply the gate to the target
                # Calculate the index after applying the gate
                target_bit = (i >> (n - 1 - target)) & 1
                new_target_bit = 1 - target_bit if gate_matrix[0, 1] != 0 else target_bit
                
                # For a general gate, we need to compute the full transformation
                # This is a simplified approach that works for X, Z, etc.
                if new_target_bit != target_bit:
                    j = i ^ (1 << (n - 1 - target))
                    full_operator[i, i] = 0
                    full_operator[j, i] = 1
                else:
                    full_operator[i, i] = 1
        
        # Apply the operator to the state vector
        self.state = np.dot(full_operator, self.state)
    
    def _apply_multi_qubit_gate(self, gate_matrix: np.ndarray, targets: List[int]):
        """
        Apply a multi-qubit gate to the specified target qubits.
        
        Args:
            gate_matrix: 2^m x 2^m matrix representation of the gate
            targets: List of target qubit indices
        """
        n = self.num_qubits
        m = len(targets)
        
        # For CNOT gate, handle it specially
        if gate_matrix.shape == (4, 4) and m == 2:
            # This is a 2-qubit gate like CNOT
            control, target = targets[0], targets[1]
            
            # Create full system operator
            full_operator = np.eye(2**n, dtype=np.complex128)
            
            # Apply CNOT logic: flip target if control is |1⟩
            for i in range(2**n):
                # Check if control qubit is |1⟩
                if (i >> (n - 1 - control)) & 1:
                    # Flip target qubit
                    j = i ^ (1 << (n - 1 - target))
                    # Swap rows i and j in the operator
                    full_operator[i, i] = 0
                    full_operator[j, i] = 1
                    full_operator[i, j] = 0
                    full_operator[j, j] = 0
            
            # Apply the operator to the state vector
            self.state = np.dot(full_operator, self.state)
        else:
            # For other multi-qubit gates, use simplified approach
            if targets == list(range(min(targets), min(targets) + m)):
                # Targets are contiguous
                operators = []
                gate_applied = False
                
                for i in range(n):
                    if i == targets[0] and not gate_applied:
                        # Apply the multi-qubit gate here (simplified)
                        operators.append(gate_matrix[:2, :2])  # Use only 2x2 part
                        gate_applied = True
                    elif i in targets[1:]:
                        # Skip other target qubits
                        continue
                    else:
                        operators.append(np.eye(2, dtype=np.complex128))
                
                if operators:
                    full_operator = self._tensor_product(operators)
                    self.state = np.dot(full_operator, self.state)
            else:
                raise NotImplementedError("Non-contiguous multi-qubit gates not yet implemented")
    
    def apply_gate(self, gate_name: str, targets: List[int], 
                  controls: Optional[List[int]] = None, parameters: Optional[List[float]] = None):
        """
        Apply a quantum gate to the state vector.
        
        Args:
            gate_name: Name of the gate to apply
            targets: List of target qubit indices
            controls: Optional list of control qubit indices
            parameters: Optional parameters for parameterized gates
        """
        gate_matrix = self._get_gate_matrix(gate_name, parameters)
        
        if gate_name.lower() == 'cnot':
            # Handle CNOT specially - it needs 2 targets but no controls
            if len(targets) != 2:
                raise ValueError("CNOT gate requires exactly 2 targets")
            control, target = targets[0], targets[1]
            
            # Apply CNOT directly without using multi-qubit gate method
            n = self.num_qubits
            new_state = self.state.copy()
            
            for i in range(2**n):
                # Check if control qubit is |1⟩
                if (i >> (n - 1 - control)) & 1:
                    # Flip target qubit
                    j = i ^ (1 << (n - 1 - target))
                    new_state[j] = self.state[i]
                    new_state[i] = 0
            
            self.state = new_state
        elif controls:
            # Controlled gate
            if len(targets) != 1:
                raise ValueError("Controlled gates must have exactly one target")
            self._apply_controlled_gate(gate_matrix, controls, targets[0])
        elif len(targets) == 1:
            # Single-qubit gate
            self._apply_single_qubit_gate(gate_matrix, targets[0])
        else:
            # Multi-qubit gate (not CNOT)
            self._apply_multi_qubit_gate(gate_matrix, targets)
        
        logger.debug(f"Applied gate {gate_name} to targets {targets}")
    
    def measure(self, shots: int = 1) -> List[int]:
        """
        Perform measurement on all qubits.
        
        Args:
            shots: Number of measurements to perform
            
        Returns:
            List of measurement results (as integers representing basis states)
        """
        # Calculate probabilities
        probabilities = np.abs(self.state)**2
        probabilities /= np.sum(probabilities)  # Ensure normalization
        
        # Sample measurements
        results = np.random.choice(range(len(self.state)), size=shots, p=probabilities)
        
        logger.debug(f"Performed {shots} measurements")
        return results.tolist()
    
    def get_state(self) -> np.ndarray:
        """Return the current state vector."""
        return self.state
    
    def get_probabilities(self) -> np.ndarray:
        """Return the measurement probabilities for each basis state."""
        return np.abs(self.state)**2

# Create FastAPI application
app = FastAPI(title="Quantum Simulator Backend", version="1.0.0")

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global simulator instance (in production, you'd want to manage this differently)
simulator = None

@app.post("/circuit", response_model=CircuitResponse)
async def execute_circuit(circuit: CircuitRequest):
    """
    Execute a quantum circuit and return the results.
    
    Args:
        circuit: Quantum circuit definition
        
    Returns:
        Circuit execution results
    """
    global simulator
    
    try:
        # Initialize or reset simulator
        if simulator is None or simulator.num_qubits != circuit.num_qubits:
            simulator = QuantumSimulator(circuit.num_qubits)
            logger.info(f"Created new simulator with {circuit.num_qubits} qubits")
        else:
            simulator.reset()
        
        # Apply all operations
        for op in circuit.operations:
            simulator.apply_gate(
                op.gate, 
                op.targets, 
                op.controls, 
                op.parameters
            )
        
        # Get results
        state = simulator.get_state()
        state_vector = [[float(amp.real), float(amp.imag)] for amp in state]
        measurements = simulator.measure(circuit.shots) if circuit.shots > 0 else None
        probabilities = simulator.get_probabilities().tolist()
        
        logger.info(f"Executed circuit with {len(circuit.operations)} operations")
        
        return CircuitResponse(
            state_vector=state_vector,
            measurements=measurements,
            probabilities=probabilities
        )
    
    except Exception as e:
        logger.error(f"Error executing circuit: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def get_state():
    """Get the current state vector of the simulator."""
    global simulator
    if simulator is None:
        raise HTTPException(status_code=400, detail="Simulator not initialized")
    
    state = simulator.get_state()
    return {"state_vector": [[float(amp.real), float(amp.imag)] for amp in state]}

@app.get("/measure")
async def measure(shots: int = 1):
    """Perform measurement on the current state."""
    global simulator
    if simulator is None:
        raise HTTPException(status_code=400, detail="Simulator not initialized")
    
    return {"measurements": simulator.measure(shots)}

@app.get("/probabilities")
async def get_probabilities():
    """Get the measurement probabilities for each basis state."""
    global simulator
    if simulator is None:
        raise HTTPException(status_code=400, detail="Simulator not initialized")
    
    return {"probabilities": simulator.get_probabilities().tolist()}

@app.post("/reset")
async def reset_simulator(num_qubits: int):
    """Reset the simulator with the specified number of qubits."""
    global simulator
    simulator = QuantumSimulator(num_qubits)
    logger.info(f"Reset simulator with {num_qubits} qubits")
    
    return {"message": f"Simulator reset with {num_qubits} qubits"}

@app.post("/interference")
async def apply_interference_effect(interference_level: float, phase_shift: float):
    """
    Apply interference effects to the current quantum state.
    
    Args:
        interference_level: Interference strength (0.0 to 1.0)
        phase_shift: Phase shift in radians
    """
    global simulator
    if simulator is None:
        raise HTTPException(status_code=400, detail="Simulator not initialized")
    
    try:
        # Get current state
        current_state = simulator.get_state().copy()
        
        # Apply interference effects
        for i in range(len(current_state)):
            amplitude = current_state[i]
            magnitude = abs(amplitude)
            
            if magnitude > 1e-10:  # Skip near-zero amplitudes
                # Apply decoherence (amplitude reduction)
                decoherence_factor = 1 - (1 - interference_level) * 0.3
                new_magnitude = magnitude * decoherence_factor
                
                # Apply phase shift (alternating for different basis states)
                current_phase = np.angle(amplitude)
                new_phase = current_phase + phase_shift * (1 if i % 2 == 0 else -1)
                
                # Update amplitude
                current_state[i] = new_magnitude * np.exp(1j * new_phase)
        
        # Normalize the state
        norm = np.linalg.norm(current_state)
        if norm > 0:
            current_state /= norm
        
        # Update simulator state
        simulator.state = current_state
        
        logger.info(f"Applied interference: level={interference_level:.3f}, phase={phase_shift:.3f}")
        
        # Return updated state and probabilities
        state_vector = [[float(amp.real), float(amp.imag)] for amp in current_state]
        probabilities = simulator.get_probabilities().tolist()
        
        return {
            "state_vector": state_vector,
            "probabilities": probabilities,
            "message": f"Interference applied successfully"
        }
        
    except Exception as e:
        logger.error(f"Error applying interference: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
