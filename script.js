// QuantumFlow Application State
const quantumFlowState = {
    qubits: 3,
    gates: [],
    circuit: [],
    step: 0,
    stateVector: { '000': { real: 1, imag: 0 } },
    animation: null,
    blochSphere: null,
    nextGateId: 0
};

// Content data for educational section
const contentData = {
    'quantum-computing': {
        title: 'Quantum Computing Fundamentals',
        body: `
            <h3>What is Quantum Computing?</h3>
            <p>Quantum computing is an area of computing focused on developing computer technology based on the principles of quantum theory, which explains the nature and behavior of energy and matter on the quantum (atomic and subatomic) level.</p>
            
            <h3>Key Principles</h3>
            <ul>
                <li><strong>Superposition:</strong> Qubits can represent both 0 and 1 simultaneously</li>
                <li><strong>Entanglement:</strong> Qubits can be correlated with each other in non-classical ways</li>
                <li><strong>Interference:</strong> Quantum states can interfere like waves, enhancing correct paths and canceling wrong ones</li>
            </ul>
            
            <h3>Quantum vs. Classical Computing</h3>
            <p>While classical computers use bits (0s and 1s), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously. This allows quantum computers to process vast amounts of information in parallel.</p>
        `
    },
    'h-gate': {
        title: 'H Gate (Hadamard Gate)',
        body: `
            <h3>What is the Hadamard Gate?</h3>
            <p>The Hadamard gate is a fundamental quantum gate that creates superposition. It maps the basis state |0⟩ to (|0⟩ + |1⟩)/√2 and |1⟩ to (|0⟩ - |1⟩)/√2.</p>
            
            <h3>Matrix Representation</h3>
            <p>The Hadamard gate is represented by the following matrix:</p>
            <p>H = 1/√2 × [[1, 1], [1, -1]]</p>
            
            <h3>Usage</h3>
            <p>The Hadamard gate is often used as the first step in many quantum algorithms to create a superposition of all possible states. It's essential for algorithms like Deutsch-Jozsa, Bernstein-Vazirani, and Grover's algorithm.</p>
        `
    },
    'superposition': {
        title: 'Quantum Superposition',
        body: `
            <h3>What is Superposition?</h3>
            <p>Superposition is a fundamental principle of quantum mechanics that allows a quantum system to be in multiple states at the same time until it is measured.</p>
            
            <h3>Classical vs Quantum</h3>
            <p>In classical computing, a bit can be either 0 or 1. In quantum computing, a qubit can be 0, 1, or any quantum superposition of these states. When measured, however, it gives only one of the possible states with certain probabilities.</p>
            
            <h3>Mathematical Representation</h3>
            <p>A qubit in superposition is represented as |ψ⟩ = α|0⟩ + β|1⟩, where α and β are complex numbers called probability amplitudes, and |α|² + |β|² = 1.</p>
            
            <h3>Importance in Quantum Computing</h3>
            <p>Superposition allows quantum computers to process a vast number of possibilities simultaneously. This parallelism is what gives quantum computers their potential advantage over classical computers for certain problems.</p>
        `
    },
    'interference': {
        title: 'Quantum Interference',
        body: `
            <h3>What is Quantum Interference?</h3>
            <p>Quantum interference is a phenomenon where the probability amplitudes of quantum states combine, leading to either constructive interference (amplification) or destructive interference (cancellation).</p>
            
            <h3>Role in Quantum Algorithms</h3>
            <p>Interference is used in quantum algorithms to amplify the probability of correct answers and cancel wrong ones. This is essential for algorithms like Grover's search and Shor's factoring algorithm.</p>
            
            <h3>Controlling Interference</h3>
            <p>By adjusting phase shifts and gate parameters, we can control interference patterns to optimize quantum computations. The controls in this simulation allow you to experiment with these concepts.</p>
        `
    },
    'applications': {
        title: 'Applications of Quantum Computing',
        body: `
            <h3>Cryptography</h3>
            <p>Quantum computers could break many current encryption methods but also enable quantum cryptography that is theoretically unhackable.</p>
            
            <h3>Drug Discovery</h3>
            <p>Quantum simulations could model complex molecular interactions at an atomic level, accelerating drug development and materials science.</p>
            
            <h3>Optimization Problems</h3>
            <p>Quantum algorithms can potentially solve complex optimization problems in logistics, finance, and machine learning more efficiently than classical computers.</p>
            
            <h3>Artificial Intelligence</h3>
            <p>Quantum computing could accelerate machine learning algorithms and enable new approaches to AI.</p>
        `
    }
};

// Function to show educational content
function showContent(contentId) {
    const content = contentData[contentId] || contentData['quantum-computing'];
    document.getElementById('content-title').innerHTML = content.title;
    document.getElementById('content-body').innerHTML = content.body;
    document.getElementById('educational-content').style.display = 'block';
    
    // Scroll to the content
    document.getElementById('educational-content').scrollIntoView({ behavior: 'smooth' });
}

// Function to hide educational content
function hideContent() {
    document.getElementById('educational-content').style.display = 'none';
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Set up interference sliders (only if they exist)
    const interferenceSlider = document.getElementById('interference-level');
    const interferenceValue = document.getElementById('interference-value');
    const phaseSlider = document.getElementById('phase-shift');
    const phaseValue = document.getElementById('phase-value');
    
    if (interferenceSlider && interferenceValue) {
        interferenceSlider.addEventListener('input', function() {
            interferenceValue.textContent = this.value + '%';
        });
    }
    
    if (phaseSlider && phaseValue) {
        phaseSlider.addEventListener('input', function() {
            phaseValue.textContent = this.value + '°';
        });
    }
    
    // Set up interference pattern sliders
    const frequencySlider = document.getElementById('wave-frequency');
    const frequencyValue = document.getElementById('frequency-value');
    const amplitudeSlider = document.getElementById('wave-amplitude');
    const amplitudeValue = document.getElementById('amplitude-value');
    const phaseDiffSlider = document.getElementById('phase-difference');
    const phaseDiffValue = document.getElementById('phase-diff-value');
    const separationSlider = document.getElementById('qubit-separation');
    const separationValue = document.getElementById('separation-value');
    
    if (frequencySlider && frequencyValue) {
        frequencySlider.addEventListener('input', function() {
            frequencyValue.textContent = parseFloat(this.value).toFixed(1);
            applyInterferenceParams();
        });
    }
    
    if (amplitudeSlider && amplitudeValue) {
        amplitudeSlider.addEventListener('input', function() {
            amplitudeValue.textContent = this.value;
            applyInterferenceParams();
        });
    }
    
    if (phaseDiffSlider && phaseDiffValue) {
        phaseDiffSlider.addEventListener('input', function() {
            phaseDiffValue.textContent = this.value + '°';
            applyInterferenceParams();
        });
    }
    
    if (separationSlider && separationValue) {
        separationSlider.addEventListener('input', function() {
            separationValue.textContent = this.value + 'px';
            applyInterferenceParams();
        });
    }
    
    // Initialize charts with D3.js
    initCharts();
    
    // Initialize interference pattern visualization
    initInterferencePattern();
    
    // Set up gate dragging
    setupDragAndDrop();
    
    // Set up button event listeners
    setupEventListeners();
    
    // Update statistics
    updateStats();
});

// Set up button event listeners
function setupEventListeners() {
    const runBtn = document.getElementById('run-btn');
    const clearBtn = document.getElementById('clear-btn');
    const addQubitBtn = document.getElementById('add-qubit');
    const removeQubitBtn = document.getElementById('remove-qubit');
    const exportBtn = document.getElementById('export-btn');
    const applyInterferenceBtn = document.getElementById('apply-interference');
    const startInterferenceBtn = document.getElementById('start-interference');
    const stopInterferenceBtn = document.getElementById('stop-interference');
    const resetInterferenceBtn = document.getElementById('reset-interference');
    const applyInterferenceParamsBtn = document.getElementById('apply-interference-params');
    const nextStepBtn = document.getElementById('next-step');
    const prevStepBtn = document.getElementById('prev-step');
    
    if (runBtn) runBtn.addEventListener('click', runSimulation);
    if (clearBtn) clearBtn.addEventListener('click', clearCircuit);
    if (addQubitBtn) addQubitBtn.addEventListener('click', addQubit);
    if (removeQubitBtn) removeQubitBtn.addEventListener('click', removeQubit);
    if (exportBtn) exportBtn.addEventListener('click', exportCode);
    if (applyInterferenceBtn) applyInterferenceBtn.addEventListener('click', applyInterference);
    if (startInterferenceBtn) startInterferenceBtn.addEventListener('click', startInterferenceAnimation);
    if (stopInterferenceBtn) stopInterferenceBtn.addEventListener('click', stopInterferenceAnimation);
    if (resetInterferenceBtn) resetInterferenceBtn.addEventListener('click', resetInterferencePattern);
    if (applyInterferenceParamsBtn) applyInterferenceParamsBtn.addEventListener('click', applyInterferenceParams);
    if (nextStepBtn) nextStepBtn.addEventListener('click', nextStep);
    if (prevStepBtn) prevStepBtn.addEventListener('click', prevStep);
}

// Initialize D3.js charts
function initCharts() {
    // Probability distribution chart
    const probData = [
        {state: '000', probability: 1.0},
        {state: '001', probability: 0.0},
        {state: '010', probability: 0.0},
        {state: '011', probability: 0.0},
        {state: '100', probability: 0.0},
        {state: '101', probability: 0.0},
        {state: '110', probability: 0.0},
        {state: '111', probability: 0.0}
    ];
    
    const probSvg = d3.select("#prob-chart")
        .append("svg")
        .attr("width", "100%")
        .attr("height", "200");
        
    const margin = {top: 20, right: 20, bottom: 30, left: 40};
    const width = 400 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;
    
    const x = d3.scaleBand()
        .domain(probData.map(d => d.state))
        .range([margin.left, width + margin.left])
        .padding(0.1);
        
    const y = d3.scaleLinear()
        .domain([0, 1])
        .range([height + margin.top, margin.top]);
        
    probSvg.selectAll("rect")
        .data(probData)
        .enter()
        .append("rect")
        .attr("x", d => x(d.state))
        .attr("y", d => y(d.probability))
        .attr("width", x.bandwidth())
        .attr("height", d => height + margin.top - y(d.probability))
        .attr("fill", "#00c6ff");
        
    probSvg.append("g")
        .attr("transform", `translate(0, ${height + margin.top})`)
        .call(d3.axisBottom(x))
        .attr("color", "#cbd5e1");
        
    probSvg.append("g")
        .attr("transform", `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(y).ticks(5).tickFormat(d => `${d * 100}%`))
        .attr("color", "#cbd5e1");
    
    // Gate usage chart
    const gateData = [
        {gate: 'H', count: 0},
        {gate: 'X', count: 0},
        {gate: 'CNOT', count: 0},
        {gate: 'RX', count: 0},
        {gate: 'RZ', count: 0}
    ];
    
    const gateSvg = d3.select("#gate-chart")
        .append("svg")
        .attr("width", "100%")
        .attr("height", "200");
        
    const xGate = d3.scaleBand()
        .domain(gateData.map(d => d.gate))
        .range([margin.left, width + margin.left])
        .padding(0.1);
        
    const yGate = d3.scaleLinear()
        .domain([0, 10])
        .range([height + margin.top, margin.top]);
        
    gateSvg.selectAll("rect")
        .data(gateData)
        .enter()
        .append("rect")
        .attr("x", d => xGate(d.gate))
        .attr("y", d => yGate(d.count))
        .attr("width", xGate.bandwidth())
        .attr("height", d => height + margin.top - yGate(d.count))
        .attr("fill", "#8e2de2");
        
    gateSvg.append("g")
        .attr("transform", `translate(0, ${height + margin.top})`)
        .call(d3.axisBottom(xGate))
        .attr("color", "#cbd5e1");
        
    gateSvg.append("g")
        .attr("transform", `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(yGate).ticks(10))
        .attr("color", "#cbd5e1");
}

// Initialize interference pattern visualization
function initInterferencePattern() {
    const canvas = document.getElementById('interference-canvas');
    if (!canvas) {
        console.error('Interference canvas not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Initialize interference pattern state
    quantumFlowState.interferencePattern = {
        canvas: canvas,
        ctx: ctx,
        width: width,
        height: height,
        animationId: null,
        time: 0,
        frequency: 3.0,
        amplitude: 50,
        phaseDifference: 0,
        separation: 150
    };
    
    // Draw initial interference pattern
    drawInterferencePattern();
}

// Draw interference pattern on canvas
function drawInterferencePattern() {
    const pattern = quantumFlowState.interferencePattern;
    if (!pattern) return;
    
    const { ctx, width, height, time, frequency, amplitude, phaseDifference, separation } = pattern;
    
    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);
    
    // Create image data for pixel manipulation
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    
    // Calculate interference pattern
    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
            const index = (y * width + x) * 4;
            
            // Two qubit sources positioned at different locations
            const source1X = width / 2 - separation / 2;
            const source1Y = height / 2;
            const source2X = width / 2 + separation / 2;
            const source2Y = height / 2;
            
            // Calculate distances from sources
            const dist1 = Math.sqrt((x - source1X) ** 2 + (y - source1Y) ** 2);
            const dist2 = Math.sqrt((x - source2X) ** 2 + (y - source2Y) ** 2);
            
            // Calculate wave amplitudes with time evolution
            const wave1 = amplitude * Math.sin(frequency * dist1 * 0.05 - time * 0.1);
            const wave2 = amplitude * Math.sin(frequency * dist2 * 0.05 - time * 0.1 + phaseDifference * Math.PI / 180);
            
            // Interference: sum of amplitudes
            const interference = wave1 + wave2;
            
            // Convert to color intensity (0-255)
            const intensity = Math.max(0, Math.min(255, 128 + interference));
            
            // Create color gradient based on interference
            if (interference > 0) {
                // Constructive interference - blue to cyan
                data[index] = 0;     // Red
                data[index + 1] = intensity; // Green
                data[index + 2] = 255; // Blue
            } else {
                // Destructive interference - red to orange
                data[index] = 255;     // Red
                data[index + 1] = Math.abs(intensity); // Green
                data[index + 2] = 0;   // Blue
            }
            data[index + 3] = Math.min(255, Math.abs(interference) * 2); // Alpha
        }
    }
    
    // Draw the interference pattern
    ctx.putImageData(imageData, 0, 0);
    
    // Draw qubit source indicators
    ctx.fillStyle = '#ffff00';
    ctx.beginPath();
    ctx.arc(width / 2 - separation / 2, height / 2, 8, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(width / 2 + separation / 2, height / 2, 8, 0, 2 * Math.PI);
    ctx.fill();
    
    // Add labels
    ctx.fillStyle = '#ffffff';
    ctx.font = '14px Arial';
    ctx.fillText('Qubit 1', width / 2 - separation / 2 - 20, height / 2 - 15);
    ctx.fillText('Qubit 2', width / 2 + separation / 2 - 20, height / 2 - 15);
}

// Set up drag and drop for quantum gates
function setupDragAndDrop() {
    const gates = document.querySelectorAll('.gate');
    const circuit = document.getElementById('circuit');
    const tooltip = document.createElement('div');
    tooltip.className = 'gate-tooltip';
    document.body.appendChild(tooltip);
    
    gates.forEach(gate => {
        gate.addEventListener('dragstart', function(e) {
            e.dataTransfer.setData('text/plain', JSON.stringify({
                type: this.getAttribute('data-gate'),
                description: this.getAttribute('data-description')
            }));
            const description = this.getAttribute('data-description');
            tooltip.textContent = description;
        });
        
        gate.addEventListener('mousemove', function(e) {
            const description = this.getAttribute('data-description');
            tooltip.textContent = description;
            tooltip.style.left = (e.pageX + 10) + 'px';
            tooltip.style.top = (e.pageY + 10) + 'px';
            tooltip.style.opacity = '1';
        });
        
        gate.addEventListener('mouseleave', function() {
            tooltip.style.opacity = '0';
        });
    });
    
    circuit.addEventListener('dragover', function(e) {
        e.preventDefault();
        // Show visual indicator of where the gate will be placed
    });
    
    circuit.addEventListener('drop', function(e) {
        e.preventDefault();
        const data = JSON.parse(e.dataTransfer.getData('text/plain'));
        const gateType = data.type;
        const qubit = findClosestQubit(e.clientY);
        
        if (qubit) {
            addGateToQubit(qubit, gateType, e.clientX);
        }
        
        tooltip.style.opacity = '0';
    });
}

// Helper function to find the closest qubit to a Y coordinate
function findClosestQubit(yPos) {
    const qubits = document.querySelectorAll('.qubit');
    let closestQubit = null;
    let minDistance = Infinity;
    
    qubits.forEach(qubit => {
        const rect = qubit.getBoundingClientRect();
        const distance = Math.abs(rect.top + rect.height/2 - yPos);
        
        if (distance < minDistance) {
            minDistance = distance;
            closestQubit = qubit;
        }
    });
    
    return closestQubit;
}

// Add a gate to a qubit
function addGateToQubit(qubit, gateType, xPos) {
    const qubitIndex = parseInt(qubit.getAttribute('data-qubit'));
    const rect = qubit.getBoundingClientRect();
    const relativeX = xPos - rect.left;
    const gateId = quantumFlowState.nextGateId++;
    
    // Create gate visual
    const gateVisual = document.createElement('div');
    gateVisual.className = 'gate-visual';
    gateVisual.textContent = gateType;
    gateVisual.setAttribute('data-gate', gateType);
    gateVisual.setAttribute('data-qubit', qubitIndex);
    gateVisual.setAttribute('data-gate-id', gateId);
    gateVisual.style.left = `${relativeX - 25}px`;
    
    // Set different colors for different gates
    switch(gateType) {
        case 'H':
            gateVisual.style.background = 'linear-gradient(135deg, #4a00e0, #8e2de2)';
            break;
        case 'X':
            gateVisual.style.background = 'linear-gradient(135deg, #ff416c, #ff4b2b)';
            break;
        case 'CX':
            gateVisual.style.background = 'linear-gradient(135deg, #11998e, #38ef7d)';
            // For CNOT gate, we need to add a control line
            addControlLine(qubitIndex, relativeX, gateId);
            break;
        default:
            gateVisual.style.background = 'linear-gradient(135deg, #00c6ff, #0072ff)';
    }
    
    // Add click event to remove gate
    gateVisual.addEventListener('click', function(e) {
        e.stopPropagation();
        this.remove();
        // Also remove any connections
        document.querySelectorAll(`.connection-line[data-gate-id="${gateId}"]`).forEach(el => el.remove());
        updateCircuitState();
    });
    
    qubit.appendChild(gateVisual);
    
    // Update the circuit state
    updateCircuitState();
}

// Add control line for CNOT gate
function addControlLine(qubitIndex, xPos, gateId) {
    // This would connect to the target qubit in a real implementation
    const line = document.createElement('div');
    line.className = 'connection-line vertical-connection';
    line.style.left = `${xPos + 25}px`;
    line.style.top = '0';
    line.style.height = '100%';
    line.setAttribute('data-gate-id', gateId);
    
    // Find the target qubit (for simplicity, we'll use the next qubit)
    const targetQubit = document.querySelector(`.qubit[data-qubit="${qubitIndex + 1}"]`);
    if (targetQubit) {
        const circuit = document.getElementById('circuit');
        circuit.appendChild(line);
        
        // Add target gate
        const targetGate = document.createElement('div');
        targetGate.className = 'gate-visual';
        targetGate.textContent = 'X';
        targetGate.setAttribute('data-gate', 'X');
        targetGate.setAttribute('data-qubit', qubitIndex + 1);
        targetGate.setAttribute('data-gate-id', gateId);
        targetGate.style.left = `${xPos - 25}px`;
        targetGate.style.background = 'linear-gradient(135deg, #11998e, #38ef7d)';
        
        targetGate.addEventListener('click', function(e) {
            e.stopPropagation();
            this.remove();
            document.querySelectorAll(`.connection-line[data-gate-id="${gateId}"]`).forEach(el => el.remove());
            document.querySelectorAll(`.gate-visual[data-gate-id="${gateId}"]`).forEach(el => el.remove());
            updateCircuitState();
        });
        
        targetQubit.appendChild(targetGate);
    }
}

// Update the circuit state
function updateCircuitState() {
    // Collect all gates from the circuit
    const gates = [];
    document.querySelectorAll('.gate-visual').forEach(gate => {
        gates.push({
            type: gate.getAttribute('data-gate'),
            qubit: parseInt(gate.getAttribute('data-qubit')),
            position: parseInt(gate.style.left),
            id: parseInt(gate.getAttribute('data-gate-id'))
        });
    });
    
    // Sort gates by position to determine circuit depth
    gates.sort((a, b) => a.position - b.position);
    
    // Update the state
    quantumFlowState.gates = gates;
    
    // Update statistics
    updateStats();
    
    // Update the generated code
    updateGeneratedCode();
}

// Update statistics display
function updateStats() {
    document.getElementById('gate-count').textContent = quantumFlowState.gates.length;
    document.getElementById('qubit-count').textContent = quantumFlowState.qubits;
    
    // Calculate circuit depth (simplified)
    const positions = quantumFlowState.gates.map(g => Math.floor(g.position / 60));
    const depth = positions.length > 0 ? Math.max(...positions) + 1 : 0;
    document.getElementById('circuit-depth').textContent = depth;
    
    // Update gate usage chart
    updateGateUsageChart();
    
    // Calculate and update fidelity (simplified)
    const fidelity = Math.max(0, 100 - quantumFlowState.gates.length * 2);
    document.getElementById('fidelity').textContent = fidelity + '%';
}

// Update the generated code based on the current circuit
function updateGeneratedCode() {
    const codeOutput = document.getElementById('code-output');
    let code = `from qiskit import QuantumCircuit, Aer, execute\nimport numpy as np\n\n# Create a quantum circuit with ${quantumFlowState.qubits} qubits\nqc = QuantumCircuit(${quantumFlowState.qubits})\n\n# Add gates\n`;
    
    // Add gates to code
    quantumFlowState.gates.forEach(gate => {
        switch(gate.type) {
            case 'H':
                code += `qc.h(${gate.qubit})\n`;
                break;
            case 'X':
                // Check if it's part of a CNOT gate
                const isCnot = quantumFlowState.gates.some(g => g.id === gate.id && g.type === 'CX');
                if (!isCnot) {
                    code += `qc.x(${gate.qubit})\n`;
                }
                break;
            case 'CX':
                // For simplicity, target is next qubit
                code += `qc.cx(${gate.qubit}, ${gate.qubit + 1})\n`;
                break;
            case 'RX':
                code += `qc.rx(np.pi/2, ${gate.qubit})\n`;
                break;
            // Add more gates as needed
        }
    });
    
    code += `qc.measure_all()\n\n# Execute the circuit\nsimulator = Aer.get_backend('qasm_simulator')\nresult = execute(qc, simulator, shots=1000).result()\ncounts = result.get_counts(qc)\nprint("Counts:", counts)`;
    
    codeOutput.textContent = code;
}

// Run simulation
async function runSimulation() {
    const gates = quantumFlowState.gates;
    if (gates.length === 0) {
        alert('Please add some gates to the circuit first!');
        return;
    }
    
    try {
        // Convert gates to backend format
        const operations = gates.map(gate => {
            // Map frontend gate names to backend gate names
            let gateName = gate.type.toLowerCase();
            if (gateName === 'cx') {
                gateName = 'cnot';
            }
            
            const operation = {
                gate: gateName,
                targets: [gate.qubit]
            };
            
            // Handle controlled gates
            if (gate.control !== undefined) {
                operation.controls = [gate.control];
            }
            
            // Handle parameterized gates
            if (gate.type === 'RX' || gate.type === 'RZ') {
                operation.parameters = [Math.PI / 4]; // Default rotation angle
            }
            
            return operation;
        });
        
        // Prepare circuit request
        const circuitRequest = {
            operations: operations,
            num_qubits: quantumFlowState.qubits,
            shots: 1024
        };
        
        // Send request to backend
        const response = await fetch('http://localhost:8000/circuit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(circuitRequest)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Backend response error:', response.status, errorText);
            throw new Error(`Backend error: ${response.status} - ${errorText}`);
        }
        
        const result = await response.json();
        console.log('Backend response:', result);
        
        // Update state vector
        if (result.state_vector) {
            updateStateVectorFromBackend(result.state_vector);
        }
        
        // Update probability chart with real results
        if (result.probabilities) {
            console.log('Updating probabilities:', result.probabilities);
            updateProbabilityChartFromBackend(result.probabilities);
        }
        
        // Display measurement results
        if (result.measurements) {
            console.log('Displaying measurements:', result.measurements);
            displayMeasurementResults(result.measurements);
        }
        
        // Generate and display code
        generateQuantumCode(gates);
        
        // Update other visualizations
        updateGateUsageChart();
        updateQuantumStateDisplay();
        
    } catch (error) {
        console.error('Simulation error:', error);
        
        // Check if it's a network error vs backend error
        if (error.message.includes('fetch') || error.message.includes('Failed to fetch')) {
            alert('Backend not available. Using mock simulation. Start the Python backend with: python quantum_backend.py');
        } else {
            console.log('Backend responded but with error:', error.message);
            // Don't show alert for backend processing errors, just log them
        }
        
        // Generate quantum code
        generateQuantumCode(gates);
        
        // Simulate results (fallback)
        const results = simulateQuantumCircuit(gates);
        
        // Format results properly
        let resultsText = `# Simulation Results (Mock)\n\n`;
        Object.entries(results)
            .sort((a, b) => b[1] - a[1])
            .forEach(([state, count]) => {
                const total = Object.values(results).reduce((sum, c) => sum + c, 0);
                const probability = count / total;
                resultsText += `|${state}⟩: ${count} counts (${(probability * 100).toFixed(1)}%)\n`;
            });
        
        document.getElementById('simulation-output').textContent = resultsText;
        
        // Update visualizations with mock data
        updateProbabilityChart();
        updateGateUsageChart();
        updateQuantumStateDisplay();
    }
}

// Generate quantum code for display
function generateQuantumCode(gates) {
    let code = `# Quantum Circuit Simulation
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import numpy as np

# Create quantum circuit with ${quantumFlowState.qubits} qubits
qc = QuantumCircuit(${quantumFlowState.qubits}, ${quantumFlowState.qubits})

`;
    
    gates.forEach(gate => {
        const qubit = gate.qubit;
        switch(gate.type) {
            case 'H':
                code += `qc.h(${qubit})  # Hadamard gate on qubit ${qubit}\n`;
                break;
            case 'X':
                code += `qc.x(${qubit})  # Pauli-X gate on qubit ${qubit}\n`;
                break;
            case 'Y':
                code += `qc.y(${qubit})  # Pauli-Y gate on qubit ${qubit}\n`;
                break;
            case 'Z':
                code += `qc.z(${qubit})  # Pauli-Z gate on qubit ${qubit}\n`;
                break;
            case 'CNOT':
                if (gate.control !== undefined) {
                    code += `qc.cx(${gate.control}, ${qubit})  # CNOT gate\n`;
                }
                break;
            case 'RX':
                code += `qc.rx(np.pi/4, ${qubit})  # RX rotation gate\n`;
                break;
            case 'RZ':
                code += `qc.rz(np.pi/4, ${qubit})  # RZ rotation gate\n`;
                break;
        }
    });
    
    code += `
# Add measurements
qc.measure_all()

# Execute the circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts(qc)

print("Measurement results:")
for state, count in counts.items():
    probability = count / 1024
    print(f"|{state}⟩: {count} counts ({probability:.3f} probability)")
`;
    
    document.getElementById('code-output').textContent = code;
}

// Update state vector from backend results
function updateStateVectorFromBackend(stateVector) {
    // Convert complex numbers from backend format [real, imag] pairs
    const newStateVector = {};
    const numQubits = quantumFlowState.qubits;
    
    stateVector.forEach((amplitude, index) => {
        const binaryState = index.toString(2).padStart(numQubits, '0');
        // Backend sends [real, imag] pairs
        newStateVector[binaryState] = {
            real: Array.isArray(amplitude) ? amplitude[0] : amplitude,
            imag: Array.isArray(amplitude) ? amplitude[1] : 0
        };
    });
    
    quantumFlowState.stateVector = newStateVector;
    console.log('Updated state vector:', newStateVector);
}

// Update probability chart from backend results
function updateProbabilityChartFromBackend(probabilities) {
    const numQubits = quantumFlowState.qubits;
    const probData = probabilities.map((prob, index) => ({
        state: index.toString(2).padStart(numQubits, '0'),
        probability: prob
    })).filter(d => d.probability > 0.001); // Only show significant probabilities
    
    console.log('Probability data for chart:', probData);
    
    // Clear and recreate the chart with new data
    d3.select("#prob-chart").selectAll("*").remove();
    
    // Create new chart
    const margin = { top: 20, right: 30, bottom: 40, left: 40 };
    const containerWidth = document.getElementById('prob-chart').clientWidth || 400;
    const width = containerWidth - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;
    
    const svg = d3.select("#prob-chart")
        .append("svg")
        .attr("width", containerWidth)
        .attr("height", 200);
    
    const g = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
    
    if (probData.length === 0) {
        g.append("text")
            .attr("x", width / 2)
            .attr("y", height / 2)
            .attr("text-anchor", "middle")
            .attr("fill", "#cbd5e1")
            .text("No significant probabilities to display");
        return;
    }
    
    const x = d3.scaleBand()
        .domain(probData.map(d => d.state))
        .range([0, width])
        .padding(0.1);
    
    const y = d3.scaleLinear()
        .domain([0, d3.max(probData, d => d.probability)])
        .range([height, 0]);
    
    // Create bars
    g.selectAll(".bar")
        .data(probData)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", d => x(d.state))
        .attr("y", d => y(d.probability))
        .attr("width", x.bandwidth())
        .attr("height", d => height - y(d.probability))
        .attr("fill", "#00c6ff");
    
    // Add probability labels on bars
    g.selectAll(".prob-label")
        .data(probData)
        .enter()
        .append("text")
        .attr("class", "prob-label")
        .attr("x", d => x(d.state) + x.bandwidth() / 2)
        .attr("y", d => y(d.probability) - 5)
        .attr("text-anchor", "middle")
        .attr("fill", "#cbd5e1")
        .attr("font-size", "12px")
        .text(d => d.probability.toFixed(3));
    
    // Add x-axis
    g.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0, ${height})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("fill", "#cbd5e1")
        .style("font-size", "12px");
    
    // Add y-axis
    g.append("g")
        .attr("class", "y-axis")
        .call(d3.axisLeft(y))
        .selectAll("text")
        .attr("fill", "#cbd5e1")
        .style("font-size", "12px");
    
    // Style axis lines
    g.selectAll(".domain, .tick line")
        .attr("stroke", "#475569");
}

// Display measurement results
function displayMeasurementResults(measurements) {
    const numQubits = quantumFlowState.qubits;
    const counts = {};
    
    // Count measurement results
    measurements.forEach(measurement => {
        const binaryState = measurement.toString(2).padStart(numQubits, '0');
        counts[binaryState] = (counts[binaryState] || 0) + 1;
    });
    
    // Generate results text
    let resultsText = `# Measurement Results (${measurements.length} shots)\n\n`;
    
    Object.entries(counts)
        .sort((a, b) => b[1] - a[1]) // Sort by count descending
        .forEach(([state, count]) => {
            const probability = count / measurements.length;
            resultsText += `|${state}⟩: ${count} counts (${(probability * 100).toFixed(1)}%)\n`;
        });
    
    document.getElementById('simulation-output').textContent = resultsText;
}

// Simple quantum circuit simulator (fallback)
function simulateQuantumCircuit(gates) {
    // This is a very simplified simulation
    // In a real implementation, you would use a proper quantum simulator library
    
    const results = {};
    const numQubits = quantumFlowState.qubits;
    const numStates = Math.pow(2, numQubits);
    
    // Initialize with all zeros having 100% probability
    let zeroState = '';
    for (let i = 0; i < numQubits; i++) zeroState += '0';
    results[zeroState] = 1000;
    
    // Apply gate effects (simplified)
    quantumFlowState.gates.forEach(gate => {
        if (gate.type === 'H') {
            // Hadamard creates superposition - apply to all existing states
            const newResults = {};
            for (const state in results) {
                if (results[state] > 0) {
                    const count = results[state];
                    // Split probability between |0⟩ and |1⟩ for this qubit
                    const state0 = state.split('');
                    const state1 = state.split('');
                    state1[gate.qubit] = state1[gate.qubit] === '0' ? '1' : '0';
                    
                    newResults[state0.join('')] = (newResults[state0.join('')] || 0) + count / 2;
                    newResults[state1.join('')] = (newResults[state1.join('')] || 0) + count / 2;
                }
            }
            Object.assign(results, newResults);
        } else if (gate.type === 'X') {
            // X gate flips the bit for all states
            const newResults = {};
            for (const state in results) {
                if (results[state] > 0) {
                    const newState = state.split('');
                    newState[gate.qubit] = newState[gate.qubit] === '0' ? '1' : '0';
                    newResults[newState.join('')] = results[state];
                }
            }
            // Replace results with flipped states
            for (const state in results) delete results[state];
            Object.assign(results, newResults);
        } else if (gate.type === 'Z') {
            // Z gate adds phase (no change in measurement probabilities for mock)
            // Keep results unchanged for simplicity
        } else if (gate.type === 'CX') {
            // CNOT gate - flip target if control is 1
            const targetQubit = gate.qubit + 1;
            if (targetQubit < numQubits) {
                const newResults = {};
                for (const state in results) {
                    if (results[state] > 0) {
                        if (state[gate.qubit] === '1') {
                            // Flip the target qubit
                            const newState = state.split('');
                            newState[targetQubit] = newState[targetQubit] === '0' ? '1' : '0';
                            newResults[newState.join('')] = results[state];
                        } else {
                            newResults[state] = results[state];
                        }
                    }
                }
                // Copy new results back
                for (const state in results) delete results[state];
                for (const state in newResults) results[state] = newResults[state];
            }
        }
        // Add more gate simulations as needed
    });
    
    return results;
}

// Update probability chart
function updateProbabilityChart(results) {
    const probData = [];
    const numQubits = quantumFlowState.qubits;
    const numStates = Math.pow(2, numQubits);
    
    // Generate all possible states
    for (let i = 0; i < numStates; i++) {
        let state = i.toString(2).padStart(numQubits, '0');
        probData.push({
            state: state,
            probability: (results[state] || 0) / 1000
        });
    }
    
    // Clear existing chart
    d3.select("#prob-chart").selectAll("*").remove();
    
    const container = d3.select("#prob-chart");
    const containerWidth = container.node().getBoundingClientRect().width || 400;
    const containerHeight = 250;
    
    const probSvg = container
        .append("svg")
        .attr("width", containerWidth)
        .attr("height", containerHeight);
        
    const margin = {top: 20, right: 20, bottom: 40, left: 50};
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;
    
    const x = d3.scaleBand()
        .domain(probData.map(d => d.state))
        .range([0, width])
        .padding(0.1);
        
    const y = d3.scaleLinear()
        .domain([0, Math.max(1, d3.max(probData, d => d.probability))])
        .range([height, 0]);
    
    const g = probSvg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
        
    // Add bars with animation
    g.selectAll("rect")
        .data(probData)
        .enter()
        .append("rect")
        .attr("x", d => x(d.state))
        .attr("y", height)
        .attr("width", x.bandwidth())
        .attr("height", 0)
        .attr("fill", d => d.probability > 0 ? "#00c6ff" : "#1e293b")
        .transition()
        .duration(750)
        .attr("y", d => y(d.probability))
        .attr("height", d => height - y(d.probability));
    
    // Add probability labels on bars
    g.selectAll(".prob-label")
        .data(probData.filter(d => d.probability > 0.01))
        .enter()
        .append("text")
        .attr("class", "prob-label")
        .attr("x", d => x(d.state) + x.bandwidth() / 2)
        .attr("y", d => y(d.probability) - 5)
        .attr("text-anchor", "middle")
        .attr("fill", "#cbd5e1")
        .attr("font-size", "10px")
        .text(d => `${(d.probability * 100).toFixed(1)}%`);
        
    // Add x-axis
    g.append("g")
        .attr("transform", `translate(0, ${height})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("fill", "#cbd5e1")
        .style("font-size", "12px");
        
    // Add y-axis
    g.append("g")
        .call(d3.axisLeft(y).ticks(5).tickFormat(d => `${(d * 100).toFixed(0)}%`))
        .selectAll("text")
        .attr("fill", "#cbd5e1")
        .style("font-size", "12px");
    
    // Style axis lines
    g.selectAll(".domain, .tick line")
        .attr("stroke", "#475569");
}

// Update gate usage chart
function updateGateUsageChart() {
    // Count gate usage
    const gateUsage = {};
    console.log('Gates in circuit:', quantumFlowState.gates);
    
    quantumFlowState.gates.forEach(gate => {
        const gateType = gate.type === 'CX' ? 'CNOT' : gate.type;
        gateUsage[gateType] = (gateUsage[gateType] || 0) + 1;
    });
    
    console.log('Gate usage counts:', gateUsage);
    
    const gateData = [
        {gate: 'H', count: gateUsage['H'] || 0},
        {gate: 'X', count: gateUsage['X'] || 0},
        {gate: 'Y', count: gateUsage['Y'] || 0},
        {gate: 'Z', count: gateUsage['Z'] || 0},
        {gate: 'CNOT', count: gateUsage['CNOT'] || 0},
        {gate: 'RX', count: gateUsage['RX'] || 0},
        {gate: 'RZ', count: gateUsage['RZ'] || 0}
    ];
    
    // Clear existing chart
    d3.select("#gate-chart").selectAll("*").remove();
    
    const container = d3.select("#gate-chart");
    const containerWidth = container.node().getBoundingClientRect().width || 400;
    const containerHeight = 250;
    
    const gateSvg = container
        .append("svg")
        .attr("width", containerWidth)
        .attr("height", containerHeight);
        
    const margin = {top: 20, right: 20, bottom: 40, left: 50};
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;
    
    const x = d3.scaleBand()
        .domain(gateData.map(d => d.gate))
        .range([0, width])
        .padding(0.1);
        
    const y = d3.scaleLinear()
        .domain([0, Math.max(10, d3.max(gateData, d => d.count))])
        .range([height, 0]);
    
    const g = gateSvg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
        
    // Add bars with animation
    g.selectAll("rect")
        .data(gateData)
        .enter()
        .append("rect")
        .attr("x", d => x(d.gate))
        .attr("y", height)
        .attr("width", x.bandwidth())
        .attr("height", 0)
        .attr("fill", d => d.count > 0 ? "#8e2de2" : "#1e293b")
        .transition()
        .duration(750)
        .attr("y", d => y(d.count))
        .attr("height", d => height - y(d.count));
    
    // Add count labels on bars
    g.selectAll(".count-label")
        .data(gateData.filter(d => d.count > 0))
        .enter()
        .append("text")
        .attr("class", "count-label")
        .attr("x", d => x(d.gate) + x.bandwidth() / 2)
        .attr("y", d => y(d.count) - 5)
        .attr("text-anchor", "middle")
        .attr("fill", "#cbd5e1")
        .attr("font-size", "12px")
        .text(d => d.count);
        
    // Add x-axis
    g.append("g")
        .attr("transform", `translate(0, ${height})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("fill", "#cbd5e1")
        .style("font-size", "12px");
        
    // Add y-axis
    g.append("g")
        .call(d3.axisLeft(y).ticks(Math.min(10, Math.max(gateData.map(d => d.count)))))
        .selectAll("text")
        .attr("fill", "#cbd5e1")
        .style("font-size", "12px");
    
    // Style axis lines
    g.selectAll(".domain, .tick line")
        .attr("stroke", "#475569");
}

// Update quantum state display
function updateQuantumStateDisplay(stateVector) {
    const display = document.getElementById('quantum-state-display');
    if (!display) return;
    
    let stateString = '|ψ⟩ = ';
    const threshold = 0.01; // Only show amplitudes above this threshold
    let terms = [];
    
    stateVector.forEach((amplitude, index) => {
        const [real, imag] = amplitude;
        const magnitude = Math.sqrt(real * real + imag * imag);
        
        if (magnitude > threshold) {
            const binaryState = index.toString(2).padStart(quantumFlowState.qubits, '0');
            const coeff = magnitude.toFixed(2);
            terms.push(`${coeff}|${binaryState}⟩`);
        }
    });
    
    if (terms.length === 0) {
        stateString += '0.00|000⟩';
    } else {
        stateString += terms.join(' + ');
    }
    
    display.textContent = stateString;
}

// Update interference pattern visualization
function updateInterferenceVisualization(level, phase) {
    const pattern = quantumFlowState.interferencePattern;
    if (!pattern) return;
    
    // Update interference parameters
    pattern.amplitude = 50 + (level * 50); // Scale amplitude with interference level
    pattern.phaseShift = phase;
    
    // Redraw the pattern with new parameters
    drawInterferencePattern();
    
    console.log(`Interference visualization updated: amplitude=${pattern.amplitude}, phase=${phase.toFixed(2)}`);
}

// Clear circuit
function clearCircuit() {
    document.querySelectorAll('.gate-visual, .connection-line').forEach(el => el.remove());
    quantumFlowState.gates = [];
    updateCircuitState();
    
    // Reset results
    document.getElementById('simulation-output').textContent = '# Results will appear here';
    
    // Reset state vector
    let zeroState = '';
    for (let i = 0; i < quantumFlowState.qubits; i++) zeroState += '0';
    quantumFlowState.stateVector = { [zeroState]: { real: 1, imag: 0 } };
    updateQuantumStateDisplay();
}

// Add qubit
function addQubit() {
    if (quantumFlowState.qubits >= 5) {
        alert("Maximum of 5 qubits allowed in this simulation");
        return;
    }
    
    const circuit = document.getElementById('circuit');
    const newQubitIndex = quantumFlowState.qubits;
    const newQubit = document.createElement('div');
    newQubit.className = 'qubit';
    newQubit.setAttribute('data-qubit', newQubitIndex);
    
    const qubitLabel = document.createElement('div');
    qubitLabel.className = 'qubit-label';
    qubitLabel.textContent = newQubitIndex;
    
    newQubit.appendChild(qubitLabel);
    circuit.appendChild(newQubit);
    
    quantumFlowState.qubits++;
    updateStats();
}

// Remove qubit
function removeQubit() {
    if (quantumFlowState.qubits <= 1) {
        alert("At least one qubit is required");
        return;
    }
    
    const circuit = document.getElementById('circuit');
    const lastQubit = circuit.querySelector(`.qubit[data-qubit="${quantumFlowState.qubits - 1}"]`);
    if (lastQubit) {
        // Remove any gates on this qubit
        lastQubit.querySelectorAll('.gate-visual').forEach(gate => {
            const gateId = gate.getAttribute('data-gate-id');
            document.querySelectorAll(`[data-gate-id="${gateId}"]`).forEach(el => el.remove());
        });
        
        circuit.removeChild(lastQubit);
        quantumFlowState.qubits--;
        updateStats();
    }
}

// Export code
function exportCode() {
    const code = document.getElementById('code-output').textContent;
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'quantum_circuit.py';
    a.click();
    
    URL.revokeObjectURL(url);
}

// Apply interference
async function applyInterference() {
    const level = document.getElementById('interference-level').value / 100;
    const phase = document.getElementById('phase-shift').value * Math.PI / 180;
    
    try {
        // Use new backend interference endpoint
        const response = await fetch('http://localhost:8000/interference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                interference_level: level,
                phase_shift: phase
            })
        });
        
        if (!response.ok) {
            throw new Error('Backend interference endpoint not available');
        }
        
        const data = await response.json();
        
        // Update quantum state display
        updateQuantumStateDisplay(data.state_vector);
        
        // Update interference pattern visualization
        updateInterferenceVisualization(level, phase);
        
        // Update probability chart with backend results
        updateProbabilityChart(data.probabilities);
        
        console.log(`Backend interference applied: Level=${level.toFixed(2)}, Phase=${phase.toFixed(2)} rad`);
        
    } catch (error) {
        console.warn('Using local interference simulation:', error.message);
        // Fallback to local simulation
        applyLocalInterference(level, phase);
    }
}

// Apply interference effects to state vector
function applyInterferenceToState(stateVector, level, phase) {
    return stateVector.map((amplitude, index) => {
        const [real, imag] = amplitude;
        const magnitude = Math.sqrt(real * real + imag * imag);
        
        if (magnitude < 1e-10) return [0, 0]; // Skip near-zero amplitudes
        
        // Apply decoherence (reduce amplitude based on interference level)
        const decoherenceFactor = 1 - (1 - level) * 0.3; // Max 30% reduction
        const newMagnitude = magnitude * decoherenceFactor;
        
        // Apply phase shift
        const currentPhase = Math.atan2(imag, real);
        const newPhase = currentPhase + phase * (index % 2 === 0 ? 1 : -1); // Alternate phase
        
        // Convert back to complex form
        const newReal = newMagnitude * Math.cos(newPhase);
        const newImag = newMagnitude * Math.sin(newPhase);
        
        return [newReal, newImag];
    });
}

// Local interference simulation fallback
function applyLocalInterference(level, phase) {
    // Create a mock superposition state for demonstration
    const numQubits = quantumFlowState.qubits;
    const stateSize = Math.pow(2, numQubits);
    const mockState = [];
    
    // Create equal superposition with interference effects
    for (let i = 0; i < stateSize; i++) {
        const baseAmplitude = 1 / Math.sqrt(stateSize);
        const phaseShift = phase * (i % 2 === 0 ? 1 : -1);
        const decoherence = level * 0.9 + 0.1; // Keep some coherence
        
        const real = baseAmplitude * Math.cos(phaseShift) * decoherence;
        const imag = baseAmplitude * Math.sin(phaseShift) * decoherence;
        
        mockState.push([real, imag]);
    }
    
    updateQuantumStateDisplay(mockState);
    updateInterferenceVisualization(level, phase);
}

// Start interference pattern animation
function startInterferenceAnimation() {
    const pattern = quantumFlowState.interferencePattern;
    if (!pattern) return;
    
    if (pattern.animationId) {
        cancelAnimationFrame(pattern.animationId);
    }
    
    function animate() {
        pattern.time += 1;
        drawInterferencePattern();
        pattern.animationId = requestAnimationFrame(animate);
    }
    
    animate();
}

// Stop interference pattern animation
function stopInterferenceAnimation() {
    const pattern = quantumFlowState.interferencePattern;
    if (pattern && pattern.animationId) {
        cancelAnimationFrame(pattern.animationId);
        pattern.animationId = null;
    }
}

// Reset interference pattern
function resetInterferencePattern() {
    const pattern = quantumFlowState.interferencePattern;
    if (!pattern) return;
    
    stopInterferenceAnimation();
    pattern.time = 0;
    pattern.frequency = 3.0;
    pattern.amplitude = 50;
    pattern.phaseDifference = 0;
    pattern.separation = 150;
    
    // Reset sliders
    document.getElementById('wave-frequency').value = 3.0;
    document.getElementById('frequency-value').textContent = '3.0';
    document.getElementById('wave-amplitude').value = 50;
    document.getElementById('amplitude-value').textContent = '50';
    document.getElementById('phase-difference').value = 0;
    document.getElementById('phase-diff-value').textContent = '0°';
    document.getElementById('qubit-separation').value = 150;
    document.getElementById('separation-value').textContent = '150px';
    
    drawInterferencePattern();
}

// Apply interference parameters
function applyInterferenceParams() {
    const pattern = quantumFlowState.interferencePattern;
    if (!pattern) return;
    
    pattern.frequency = parseFloat(document.getElementById('wave-frequency').value);
    pattern.amplitude = parseInt(document.getElementById('wave-amplitude').value);
    pattern.phaseDifference = parseInt(document.getElementById('phase-difference').value);
    pattern.separation = parseInt(document.getElementById('qubit-separation').value);
    
    drawInterferencePattern();
}

// Next step
function nextStep() {
    if (quantumFlowState.step < quantumFlowState.gates.length) {
        quantumFlowState.step++;
        // In a real implementation, this would show the circuit state after each step
        alert(`Step ${quantumFlowState.step} of ${quantumFlowState.gates.length}`);
    }
}

// Previous step
function prevStep() {
    if (quantumFlowState.step > 0) {
        quantumFlowState.step--;
        // In a real implementation, this would show the circuit state after each step
        alert(`Step ${quantumFlowState.step} of ${quantumFlowState.gates.length}`);
    }
}