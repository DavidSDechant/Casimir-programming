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
from itertools import product

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

#---------------------------------------------------------------------------------

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
    for i in range(len(QuantumCircuit)): #going through the circuit and stepwise calculating the circuit matrix
        if QuantumCircuit[i][0] == 'h': #In case a Hadamard gate will be applied at one of the qubits
            Circuitmatrix = hadamard(QuantumCircuit, nr_qubits, initial, i) @ Circuitmatrix
        elif QuantumCircuit[i][0] == 'nop':     
            continue
        elif QuantumCircuit[i][0] == 'cnot':  
            Circuitmatrix = cnot(QuantumCircuit, nr_qubits, initial, i) @ Circuitmatrix  
            
        #elif QuantumCircuit[i:0] == 'cnot' or 'c-x':
        #    Circuitmatrix = cnot() * Circuitmatrix
        else:
            continue
    return Circuitmatrix

#---------------------------------------------------------------------------------

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


#---------------------------------------------------------------------------------

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
        wq1, wq2:       np.array
                        number of the wanted qubits as integers
    """
    wq1 = ""
    wq2 = ""
    for c in QuantumCircuit[i][1]: #reading the integer in the entry of the form q0
        if c.isdigit() and wq1 == "":
            wq1 = wq1 + c
        elif c.isdigit() and wq1 != "":
            wq2 = wq2 + c
    return np.array([int(wq1), int(wq2)])  

#---------------------------------------------------------------------------------

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
    # print("hadamard",circuit_state)
    return circuit_state


#---------------------------------------------------------------------------------

# def nop(QuantumCircuit, nr_qubits, initial, i):
#     return QuantumCircuit

#---------------------------------------------------------------------------------

def cnot(QuantumCircuit, nr_qubits, initial, i):
    """ This function creates a matrix which is applying the CNOT-gate at the qubits which are specified at the ith row in QuantumCircuit, and identity on the other qubits.
        Only works for nearest neighbour CNOT matrices!
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
                        matrix which is applying the CNOT gate at the qubits which are specified at the ith row in QuantumCircuit, and identity on the others
    """
    print('lol')
    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    cxrev = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    controlqubit = whichqubit2(QuantumCircuit, i)[0] #determining which qubit is the control qubit
    targetqubit = whichqubit2(QuantumCircuit, i)[1] # determining which qubit the x gate should be applied to
    circuit_state = []
    j = 0
    print('control is', controlqubit)
    print('target is', targetqubit)
    print('still here')
    if controlqubit < targetqubit: # we have to apply different CNOT matrices, depending on where the control and target are.
        print('im in one')
        print('nr qubits is', nr_qubits)
        while j < nr_qubits:
            print('j is', j)
            if j == controlqubit and len(circuit_state) == 0:
                circuit_state = cx
                j += 2
                print('1')
            elif len(circuit_state) == 0:         
                circuit_state = np.identity(2)
                j += 1
                print('2')
            elif j == controlqubit:
                circuit_state = np.kron(circuit_state, cx)
                j += 2
                print('3')
            else:        
                circuit_state = np.kron(circuit_state, np.identity(2))
                j += 1
                print('4')
                
    else:
        print('im in two')
        while j < nr_qubits:
            if j == targetqubit and len(circuit_state) == 0:
                circuit_state = cxrev
                j += 2
            elif len(circuit_state) == 0:         
                circuit_state = np.identity(2)
                j += 1
            elif j == targetqubit:
                circuit_state = np.kron(circuit_state, cxrev)
                j += 2
            else:        
                circuit_state = np.kron(circuit_state, np.identity(2))
                j += 1
    print(circuit_state)
    return circuit_state

            
#------------------------------------------------------------------------------------------------

def plot_probabilities(final_state,nr_qubits):
    states = np.array(list(product(range(2), repeat=nr_qubits)))
    bars = []
    for i in states[:]:
        i.astype(str)
        dec =''.join(i.astype(str))
        bars.append(dec)

    y_pos = np.arange(len(bars))

    plt.bar(y_pos,final_state**2)
    plt.xticks(y_pos, bars)
    plt.ylabel("Probabilities")
    plt.xlabel("Quantum state")

#------------------------------------------------------------------------------------------------



#-----------------------main_program-------------------------------------------------------------

QuantumCircuit = np.loadtxt("QASM-samples/test4.qasm", dtype="str") #loads the circuit from the qasm file to a 2*N matrix
nr_qubits = collections.Counter(QuantumCircuit[:,0])["qubit"]   # Gives number of qubits in circuit  
print("Data from .qasm file \n", QuantumCircuit)

initial_state = initial_quantumstate(QuantumCircuit, nr_qubits) #write the initial quantum state as a vector
print("Initial quantum state for system in ground state: \n",initial_state)  

Circuit = all_gates(QuantumCircuit, nr_qubits, initial_state) #write the circuit as a matrix
print("Circuit_matrix: \n",Circuit)

final_state = Circuit @ initial_state # Compute the quantum state after applying all gates
print("Final quantum state: \n", final_state)

# plot_probabilities(final_state,nr_qubits) # Plot the probability distribution of the final state

#-------------------------------------------------------------------------------------------------



# +
import numpy as np

# print("this will be a cnot")
# cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# print(cx)
# cxtensor = np.reshape(cx, (2,2,2,2))
# print(cxtensor)
# cxback = np.reshape(cxtensor, (4,4))
# print(cxback)
nr_qubits = 4
circuitstate = np.array([[[1,0],[0,0]]]*nr_qubits)
c = []
for i in range(nr_qubits):
    if i == 0:
        c = np.identity(2)
    else:
        c = np.kron(c, circuitstate[i])
print(c)
# circuitstate = np.reshape(circuitstate, (2**nr_qubits, 2**nr_qubits))
# circuitstate = np.zeros((2,)*nr_qubits) #creating an identity which 
# circuitstate[(0,)*nr_qubits] = 1
# # circuitstate = np.reshape(circuitstate, (2**nr_qubits, 2**nr_qubits))
# print(circuitstate)
# c = np.reshape(circuitstate, (2**nr_qubits, 2**nr_qubits))
# print(c)

    # controlqubit = whichqubit(QuantumCircuit, i) #determining which qubit is the control qubit
    # targetqubit = whichqubit2(QuantumCircuit, i) # determining which qubit the x gate should be applied to
    # circuit_state = []
    # for j in range(nr_qubits):
    #     if j == controlqubit and len(circuit_state) == 0:
    #         circuit_state = h
    #     elif len(circuit_state) == 0:         
    #         circuit_state = np.identity(2)
    #     elif j == qubitnumber:
    #         circuit_state = np.kron(circuit_state, h)
    #     else:        
    #         circuit_state = np.kron(circuit_state, np.identity(2))
    #     return circuit_state


# +
#different ansatz for general cnot, doesnt work yet, since i didnt figure out how to translate from the braket-form to a density matrix
def cnot(QuantumCircuit, nr_qubits, initial, i):
    print("this will be a cnot")
    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    cxtensor = np.reshape(cx, (2,2,2,2)) # writing the cnot gate in braket-form
    controlqubit = whichqubit(QuantumCircuit, i) #determining which qubit is the control qubit
    targetqubit = whichqubit2(QuantumCircuit, i) # determining which qubit the x gate should be applied to
    circuitstate = np.zeros((2,)*nr_qubits) #creating an identity which 
    circuitstate[(0,)*n] = 1
    #circuit_state = []
    # for j in range(nr_qubits):
    #     if j == controlqubit and len(circuit_state) == 0:
    #         circuit_state = h
    #     elif len(circuit_state) == 0:         
    #         circuit_state = np.identity(2)
    #     elif j == qubitnumber:
    #         circuit_state = np.kron(circuit_state, h)
    #     else:        
    #         circuit_state = np.kron(circuit_state, np.identity(2))
    #     return circuit_state

            
