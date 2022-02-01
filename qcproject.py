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
#--------------------------packages------------------------------------------------
import numpy as np
import collections #package used for counting the number of qubits in the circuit

#--------------------------functions------------------------------------------------

def initial_quantumstate(QuantumCircuit, nr_qubits):
    
    """This function determines the inital quantum state when all qubits are in the ground state.

    Input
    -----
        QuantumCircuit: np.array
                        The whole quantum circuit
        nr_qubits:      int
                        number of qubits in the circuit

    Output
    -----
        qubit_state:    np.array 
                        initual quantum state
    """
    
    qubit_state = []   # define empty array 
    
    for i in range(nr_qubits):
        if len(qubit_state) == 0:         
            qubit_state = np.array([1,0])
        else:        
            qubit_state = np.kron(qubit_state, np.array([1,0]))
            
    return qubit_state


def all_gates(QuantumCircuit, nr_qubits, initial):
    """This function determines the matrix which corresponds to applying the whole circuit.
    
    Input
    -----
        QuantumCircuit: np.array
                        The quantum circuit in the form of a numpy array
        nr_qubits:      int
                        number of qubits in the circuit
        initial:        np.array
                        The initial quantum state
    Output
    -----
        Circuitmatrix:  np.array
                        The matrix which corresponds to applying the whole circuit
    """
    Circuitmatrix = np.identity(len(initial))
    for i in range(len(QuantumCircuit)):#going through the circuit and stepwise calculating the circuit matrix
        if QuantumCircuit[i][0] == 'h': #In case a Hadamard gate will be applied at one of the qubits
            Circuitmatrix = hadamard(QuantumCircuit, nr_qubits, initial, i) @ Circuitmatrix
        #elif QuantumCircuit[i:0] == 'cnot' or 'c-x':
        #    Circuitmatrix = cnot() * Circuitmatrix
        else:
            continue
    return Circuitmatrix


def whichqubit(QuantumCircuit, i):
    """This function determines at which qubit we have to apply the 1-qubit-gate, basically reading which qubit is written at the ith row, second column in the QuantumCircuit.
    
    Input
    -----
        QuantumCircuit: np.array
                        The quantum circuit in the form of a numpy array
        i:              int
                        The row in quantum Circuit, where the second entry is written in the form q0, which is the wanted qubit
    Output
    -----
        wq:             int
                        number of the wanted qubit as an integer
    """
    wq = ""
    for c in QuantumCircuit[i][1]: #reading the integer in the entry of the form q0
        if c.isdigit():
            wq = wq + c
    return int(wq)

def whichqubit2(QuantumCircuit, i):
    """This function determines at which qubits we have to apply the 2-qubit-gate, basically reading which qubits are written at the ith row, second column in the QuantumCircuit.
    
    Input
    -----
        QuantumCircuit: np.array
                        The quantum circuit in the form of a numpy array
        i:              int
                        The row in quantum Circuit, where the second entry is written in the form q0, q1, which are the wanted qubits
    Output
    -----
        wq:             np.array
                        number of the wanted qubits as integers
    """
    wq1 = ""
    wq2 = ""
    for c in QuantumCircuit[i][1]: #reading the integer in the entry of the form q0
        if c.isdigit() and wq1 == "":
            wq1 = wq1 + c
        if c.isdigit() and wq1 != "":
            wq2 = wq2 + c
    return np.array([int(wq1), int(wq2)]) 


def hadamard(QuantumCircuit, nr_qubits, initial, i):
    """This function creates a matrix which is applying the Hadamard gate at the qubit which is specified at the ith row in QuantumCircuit, and identity on the other qubits.

    Input
    -----
        QuantumCircuit: np.array
                        The whole quantum circuit
        nr_qubits:      int
                        number of qubits in the circuit
        initial:        np.array
                        The initial quantum state
        i:              int
                        The row in quantum Circuit, where the second entry is written in the form q0, which is the wanted qubit

    Output
    -----
        circuit_state:  np.array 
                        matrix which is applying the Hadamard gate at the qubit which is specified at the ith row in QuantumCircuit, and identity on the other qubits.
    """
    h = np.array([[1, 1], [1, -1]])/np.sqrt(2) #Hadamard gate acting on one qubit
    qubitnumber = whichqubit(QuantumCircuit, i) #determining the qubit where we apply the Hadamard gate
    circuit_state = []
    for j in range(nr_qubits): #building the matrix by checking for each qubit if we should apply Hadamard or identity
        if j == qubitnumber and len(circuit_state) == 0: 
            circuit_state = h
        elif len(circuit_state) == 0:         
            circuit_state = np.identity(2)
        elif j == qubitnumber:
            circuit_state = np.kron(circuit_state, h) #tensor product of the current matrix with Hadamard
        else:        
            circuit_state = np.kron(circuit_state, np.identity(2)) 
    return circuit_state
            
#------------------------------------------------------------------------------------------------


#-----------------------main_program-----------------------------------------------------------
QuantumCircuit = np.loadtxt("QASM-samples/test3.qasm", dtype="str") #loads the circuit from the qasm file to a 2*N matrix
nr_qubits = collections.Counter(QuantumCircuit[:,0])["qubit"]   # Gives number of qubits in circuit  
print(QuantumCircuit)

initial = initial_quantumstate(QuantumCircuit, nr_qubits) #write the initial quantum state as a vector
print(initial)  

Circuit = all_gates(QuantumCircuit, nr_qubits, initial) #write the circuit as a matrix
print(Circuit)
#-----------------------------------------------------------------------------------------------
# -

import numpy as np
QuantumCircuit = np.loadtxt("QASM-samples/test1.qasm", dtype="str")
print(QuantumCircuit[0][1])
  wq = ""
    for c in QuantumCircuit[i:1]:
        if c.isdigit():
            wq = wq + c
    return int(wq)


