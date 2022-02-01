# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import collections

#--------------------------functions------------------------------------------------

def initial_quantumstate(QuantumCircuit, nr_qubits):
    
    """This function determines the inital quantum state when all qubits are in the ground state.

    Input
    -----
        QuantumCircuit: The quantum circuit in the form of a numpy array

    Output
    -----
        numpy array of the initual quantum state
    """
    
    qubit_state = []   # define empty array 
    
    for i in range(nr_qubits):
        if len(qubit_state) == 0:         
            qubit_state = np.array([1,0])
        else:        
            qubit_state = np.kron(qubit_state, np.array([1,0]))
            
    return qubit_state

#---------------------------------------------------------------------------------

def all_gates(QuantumCircuit, nr_qubits, initial):
    """This function determines the matrix which corresponds to applying the whole circuit.
    
    Input
    -----
        QuantumCircuit: 2*N dimensional np.array
                        The quantum circuit in the form of a numpy array
        initial:        np.array
                        The initial quantum state
    Output
    -----
        (dim(initial))**2 dimensional np.array
        The matrix which corresponds to applying the whole circuit
    """
    Circuitmatrix = np.identity(len(initial))
    print(Circuitmatrix)
    for i in range(len(QuantumCircuit)):#going through the circuit and stepwise calculating the circuit matrix
        if QuantumCircuit[i][0] == 'h':
            Circuitmatrix = hadamard(QuantumCircuit, nr_qubits, initial, i) @ Circuitmatrix
        elif QuantumCircuit[i][0] == 'nop':     
            Circuitmatrix = nop(QuantumCircuit, nr_qubits, initial, i) @ Circuitmatrix
        elif QuantumCircuit[i][0] == 'cnot':  
            print(QuantumCircuit[i][0])
            # Circuitmatrix = cnot(QuantumCircuit, nr_qubits, initial, i) @ Circuitmatrix    
            
        #elif QuantumCircuit[i:0] == 'cnot' or 'c-x':
        #    Circuitmatrix = cnot() * Circuitmatrix
        else:
            continue
    return Circuitmatrix

#---------------------------------------------------------------------------------

def whichqubit(QuantumCircuit, i):
    """This function determines at which qubit we have to apply the gate, 
     basically reading which qubit is written at the i'th row, second column in the QuantumCircuit.
    
    Input
    -----
        QuantumCircuit: 2*N dimensional np.array
                        The quantum circuit in the form of a numpy array
        i:              int
                        The row in quantum Circuit, where the second entry is written in the form q0, which is the wanted qubit
    Output
    -----
        wq: int
            number of the wanted qubit as an integer
    """
    wq = ""
    for c in QuantumCircuit[i][1]: #reading the integer in the entry of the form q0
        if c.isdigit():
            wq = wq + c
    return int(wq)

#---------------------------------------------------------------------------------

def hadamard(QuantumCircuit, nr_qubits, initial, i):
    h = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    qubitnumber = whichqubit(QuantumCircuit, i)
    circuit_state = []
    for j in range(nr_qubits):
        if j == qubitnumber and len(circuit_state) == 0:
            circuit_state = h
        elif len(circuit_state) == 0:         
            circuit_state = np.identity(2)
        elif j == qubitnumber:
            circuit_state = np.kron(circuit_state, h)
        else:        
            circuit_state = np.kron(circuit_state, np.identity(2))
    return circuit_state


#---------------------------------------------------------------------------------

def nop(QuantumCircuit, nr_qubits, initial, i):
    return circuit_state

#---------------------------------------------------------------------------------

def cnot(QuantumCircuit, nr_qubits, initial, i):
    print("this will be a cnot")
    # h = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    # qubitnumber = whichqubit(QuantumCircuit, i)
    # circuit_state = []
    # for j in range(nr_qubits):
    #     if j == qubitnumber and len(circuit_state) == 0:
    #         circuit_state = h
    #     elif len(circuit_state) == 0:         
    #         circuit_state = np.identity(2)
    #     elif j == qubitnumber:
    #         circuit_state = np.kron(circuit_state, h)
    #     else:        
    #         circuit_state = np.kron(circuit_state, np.identity(2))
    # return circuit_state

            
#------------------------------------------------------------------------------------------------


#-----------------------main_program-----------------------------------------------------------
QuantumCircuit = np.loadtxt("QASM-samples/test2.qasm", dtype="str") #loads the circuit from the qasm file to a 2*N matrix
nr_qubits = collections.Counter(QuantumCircuit[:,0])["qubit"]   # Gives number of qubits in circuit  

initial = initial_quantumstate(QuantumCircuit, nr_qubits) #write the initial quantum state as a vector
print(initial)  

Circuit = all_gates(QuantumCircuit, nr_qubits, initial) #write the circuit as a matrix
print(Circuit)

