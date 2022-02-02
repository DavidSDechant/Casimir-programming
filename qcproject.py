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

# # +
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
    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) #cnot gate on two qubits, control is 0, target is 1
    cxrev = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]) #cnot gate on two qubits, control is 1, target 0
    controlqubit = whichqubit2(QuantumCircuit, i)[0] #determining which qubit is the control qubit
    targetqubit = whichqubit2(QuantumCircuit, i)[1] # determining which qubit the x gate should be applied to
    circuit_state = []
    j = 0
    if controlqubit < targetqubit: # we have to apply different CNOT matrices, depending on where the control and target are.
        while j < nr_qubits: #building the whole circuit matrix, where cnot is at the correct spot, identity else
            if j == controlqubit and len(circuit_state) == 0: #if cnot is at beginning
                circuit_state = cx
                j += 2
            elif len(circuit_state) == 0:         #if cnot is not at beginning
                circuit_state = np.identity(2)
                j += 1
            elif j == controlqubit:          #here we insert cnot
                circuit_state = np.kron(circuit_state, cx)
                j += 2
            else:        #identity for the rest of the matrices
                circuit_state = np.kron(circuit_state, np.identity(2))
                j += 1
                
    else:
        while j < nr_qubits:  #building the whole circuit matrix, where cnot is at the correct spot, identity else
            if j == targetqubit and len(circuit_state) == 0: #if cnot is at beginning
                circuit_state = cxrev
                j += 2
            elif len(circuit_state) == 0:       #if cnot is not at beginning  
                circuit_state = np.identity(2)
                j += 1
            elif j == targetqubit:   #here we insert cnot
                circuit_state = np.kron(circuit_state, cxrev)
                j += 2 
            else:          #identity for the rest of the matrices
                circuit_state = np.kron(circuit_state, np.identity(2))
                j += 1
    return circuit_state

            
#------------------------------------------------------------------------------------------------

def plot_probabilities(final_state,nr_qubits):
    """This function doesn't work yet, is not used in main
    """
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

QuantumCircuit = np.loadtxt("QASM-samples/test5.qasm", dtype="str") #loads the circuit from the qasm file to a 2*N matrix
nr_qubits = collections.Counter(QuantumCircuit[:,0])["qubit"]   # Gives number of qubits in circuit  
print("Data from .qasm file \n", QuantumCircuit)

initial_state = initial_quantumstate(QuantumCircuit, nr_qubits) #write the initial quantum state as a vector
print("Initial quantum state for system in ground state: \n",initial_state)  

Circuit = all_gates(QuantumCircuit, nr_qubits, initial_state) #write the circuit as a matrix
print("Circuit_matrix: \n",Circuit)

final_state = Circuit @ initial_state # Compute the quantum state after applying all gates
print("Final quantum state: \n", final_state)

plot_probabilities(final_state,nr_qubits) # Plot the probability distribution of the final state

#-------------------------------------------------------------------------------------------------
# -



# +
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

N = 1000 # Number of shots    
    
final_state 

projectors=[np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ] # arrray containing the projectors |0><0| and |1><1|
projectors[0]



# def project(i,j,reg): # RETURN state with ith qubit of reg projected onto |j>
#     projected=np.tensordot(projectors[j],reg.psi,(1,i))
#     return np.moveaxis(projected,0,i)

# project(i,j,final)

# from scipy.linalg import norm 

def measure(i,reg): 
    projectors=[ np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ] 
    
    def project(i,j,reg): 
        projected=np.tensordot(projectors[j],reg.psi,(1,i))
        return np.moveaxis(projected,0,i)
    
    projected=project(i,0,reg) 
    norm_projected=norm(projected.flatten()) 
    if np.random.random()<norm_projected**2: 
        reg.psi=projected/norm_projected
        return 0
    else:
        projected=project(i,1,reg)
        reg.psi=projected/norm(projected)
        return 1
    
    

# +
import numpy as np
from scipy.linalg import norm 

H_matrix=1/np.sqrt(2)*np.array([[1, 1],
                                [1,-1]])

CNOT_matrix=np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,0,1],
                      [0,0,1,0]])

CNOT_tensor=np.reshape(CNOT_matrix, (2,2,2,2))

class Reg: 
    def __init__(self,n):
        self.n=n
        self.psi=np.zeros((2,)*n) 
        self.psi[(0,)*n]=1
        
def H(i,reg): 
    reg.psi=np.tensordot(H_matrix,reg.psi,(1,i)) 
    reg.psi=np.moveaxis(reg.psi,0,i)

def CNOT(control, target, reg):
    reg.psi=np.tensordot(CNOT_tensor, reg.psi, ((2,3),(control, target))) 
    reg.psi=np.moveaxis(reg.psi,(0,1),(control,target))   

def measure(i,reg): 
    projectors=[ np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) ] 
    
    def project(i,j,reg): 
        projected=np.tensordot(projectors[j],reg.psi,(1,i))
        return np.moveaxis(projected,0,i)
    
    projected=project(i,0,reg) 
    norm_projected=norm(projected.flatten()) 
    if np.random.random()<norm_projected**2: 
        reg.psi=projected/norm_projected
        return 0
    else:
        projected=project(i,1,reg)
        reg.psi=projected/norm(projected)
        return 1
    
# Example of final usage: create uniform superposition
reg=Reg(4)
for i in range(reg.n):
    H(i,reg)
    
print(reg.psi.flatten())
# -



# +
def basis_states(nr_qubits):
    qubit_state = []   # define empty array 
    
    for i in range(nr_qubits):
        if len(qubit_state) == 0:         
            qubit_state = np.array([1,0])
        else:        
            qubit_state = np.kron(qubit_state, np.array([1,0]))        
    return qubit_state

basis_states(2)

[int(d) for d in str(n)]





# +

# Python3 code to demonstrate
# getting numbers from string 
# using List comprehension + isdigit() +split()
  
# initializing string 
test_string = "There are 23 apples for 4 persons"
  
# printing original string 
print("The original string : " + test_string)
  
# using List comprehension + isdigit() +split()
# getting numbers from string 
res = [int(i) for i in test_string.split() if i.isdigit()]
  
# print result
print("The numbers list is : " + str(res))
# -

initial_quantumstate(QuantumCircuit, 2)

# +
states = np.array(list(product(range(2), repeat=nr_qubits)))
bars = []
state_list = []
for i in states[:]:
    i.astype(str)
    dec =''.join(i.astype(str))
    bars.append(dec)   # May have to reverse the order of bars

for index in bars:
    label = [int(d) for d in str(index)]
    print(label)
    full_state = []
    for state in label:
        if state == 0: 
            qubit_state = np.array([1,0])
        else:
            qubit_state = np.array([0,1])
        
        if np.size(full_state) == 0:
            full_state = qubit_state
        else:
            full_state = np.kron(full_state, qubit_state)
        
    print(full_state)    
        
    
    state_list = np.append(state_list,full_state)
state_list = np.reshape(state_list,(2**nr_qubits,4))    
print(state_list)    
# -


