# QuantumFlow - Interactive Quantum Computing Simulator

A comprehensive quantum computing visualization and simulation platform with real-time interference pattern visualization and a powerful Python backend.

## Features

### Frontend (Web Interface)
- **Interactive Quantum Circuit Builder**: Drag-and-drop quantum gates onto qubits
- **Real-time Quantum Interference Patterns**: Canvas-based visualization with adjustable parameters
- **Quantum Gate Library**: H, X, Y, Z, CNOT, RX, RZ gates with visual representations
- **Live Probability Charts**: D3.js-powered visualization of quantum state probabilities
- **Gate Usage Statistics**: Track and visualize gate usage in circuits
- **Educational Content**: Interactive learning modules about quantum computing concepts
- **Code Generation**: Automatic Qiskit code generation for circuits

### Backend (Python Quantum Simulator)
- **High-Performance State Vector Simulation**: NumPy-based quantum state simulation
- **Comprehensive Gate Support**: Single-qubit, multi-qubit, and parameterized gates
- **RESTful API**: FastAPI-based backend with automatic documentation
- **Real Quantum Measurements**: Statistical sampling from quantum probability distributions
- **Scalable Architecture**: Support for up to 25 qubits (practical simulation limit)

## Quick Start

### 1. Frontend Setup
Simply open `index.html` in your web browser:
```bash
# Open the file directly or serve it locally
python -m http.server 8080  # Optional: serve on localhost:8080
```

### 2. Backend Setup (Optional but Recommended)

#### Install Python Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

#### Start the Quantum Simulator Backend
```bash
# Run the FastAPI server
python quantum_backend.py
```

The backend will start on `http://localhost:8000` with automatic API documentation at `http://localhost:8000/docs`.

### 3. Using the Application

1. **Build Quantum Circuits**: Drag gates from the palette onto the circuit qubits
2. **Adjust Parameters**: Use sliders to modify interference pattern parameters
3. **Run Simulations**: Click "Run Simulation" to execute circuits
4. **View Results**: See real-time probability distributions and measurement results
5. **Export Code**: Generate Qiskit Python code for your circuits

## Architecture

### Frontend Components
- `index.html` - Main application interface
- `script.js` - Application logic and backend integration
- `style.css` - Modern UI styling with quantum-themed design

### Backend Components
- `quantum_backend.py` - FastAPI server with quantum simulator
- `requirements.txt` - Python dependencies

## API Endpoints

### POST /circuit
Execute a quantum circuit with specified operations.

**Request Body:**
```json
{
  "operations": [
    {
      "gate": "h",
      "targets": [0],
      "controls": null,
      "parameters": null
    }
  ],
  "num_qubits": 3,
  "shots": 1024
}
```

**Response:**
```json
{
  "state_vector": [...],
  "measurements": [...],
  "probabilities": [...]
}
```

### GET /state
Get the current quantum state vector.

### GET /probabilities
Get measurement probabilities for all basis states.

### POST /reset
Reset the simulator with a specified number of qubits.

## Supported Quantum Gates

### Single-Qubit Gates
- **I**: Identity gate
- **X**: Pauli-X (NOT) gate
- **Y**: Pauli-Y gate
- **Z**: Pauli-Z gate
- **H**: Hadamard gate (creates superposition)
- **S**: S gate (phase gate)
- **T**: T gate (π/8 gate)

### Parameterized Gates
- **RX(θ)**: Rotation around X-axis
- **RY(θ)**: Rotation around Y-axis
- **RZ(θ)**: Rotation around Z-axis
- **U(θ,φ,λ)**: General single-qubit unitary

### Multi-Qubit Gates
- **CNOT**: Controlled-NOT gate
- **CZ**: Controlled-Z gate
- **SWAP**: Swap gate

## Interference Pattern Visualization

The application features a unique quantum interference pattern visualization that demonstrates:

- **Wave Interference**: Visual representation of quantum wave behavior
- **Phase Relationships**: Adjustable phase differences between qubits
- **Constructive/Destructive Interference**: Color-coded visualization
- **Real-time Parameter Control**: Interactive sliders for frequency, amplitude, and separation

### Controls
- **Frequency**: Wave frequency (1-10)
- **Amplitude**: Wave strength (10-100)
- **Phase Difference**: Phase relationship (0-360°)
- **Qubit Separation**: Distance between sources (50-300px)

## Educational Content

Interactive learning modules covering:
- Quantum Computing Fundamentals
- Quantum Gates and Operations
- Superposition and Entanglement
- Quantum Interference
- Quantum Algorithms

## Technical Requirements

### Frontend
- Modern web browser with JavaScript ES6+ support
- Canvas API support for interference visualization
- D3.js for data visualization

### Backend
- Python 3.8+
- NumPy 1.24+
- FastAPI 0.104+
- SciPy 1.11+
- Uvicorn for ASGI server

## Development

### Running in Development Mode
```bash
# Frontend: Serve with live reload
python -m http.server 8080

# Backend: Run with auto-reload
uvicorn quantum_backend:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Gates
1. Add gate matrix to `SINGLE_QUBIT_GATES` in `quantum_backend.py`
2. Update gate handling in `_get_gate_matrix()` method
3. Add frontend representation in `script.js`
4. Update UI in `index.html`

## Performance Notes

- **State Vector Simulation**: Exponential memory scaling (2^n complex numbers)
- **Practical Limit**: ~25 qubits on typical hardware
- **Optimization**: Uses NumPy for vectorized operations
- **Memory Usage**: ~1GB for 20 qubits, ~32GB for 25 qubits

## Troubleshooting

### Backend Connection Issues
If you see "Backend not available" messages:
1. Ensure Python backend is running: `python quantum_backend.py`
2. Check that port 8000 is not blocked
3. Verify all dependencies are installed: `pip install -r requirements.txt`

### Performance Issues
- Reduce number of qubits for faster simulation
- Decrease number of shots for quicker results
- Close other applications to free memory

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**QuantumFlow** - Bringing quantum computing to everyone through interactive visualization and simulation.
