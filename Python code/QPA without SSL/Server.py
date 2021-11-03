from qiskit import (
    IBMQ,
    QuantumCircuit,
    execute,
    Aer,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit.ignis.mitigation.measurement import complete_meas_cal, tensored_meas_cal, CompleteMeasFitter, TensoredMeasFitter
import numpy as np
import math
import random


class Server():
    def __init__(self):
        print('Server initialized')

    def generate_point(self, size=14):
        """
        Generate a random point as password

        Parameters:
        size (int): number of qubits in circuit to be encrypted

        Returns:
        point (str): the combination of key1 and key2 in binary form
        key1 ([int]): permutation key in integer form
        key2 ([[int], [int]]): one-time pad component of the password, comprised of the x key and z key respectively
        """
        # get permutation key
        bin_key1, key1 = self.generate_permutation_key()
        # get OTP key
        x_key = np.random.randint(2, size=size)
        z_key = np.random.randint(2, size=size)
        key2 = [x_key.tolist(), z_key.tolist()]
        # combine keys to get point
        a = ''.join(bin_key1)
        b = ''.join(map(str, x_key.tolist()))
        c = ''.join(map(str, z_key.tolist()))
        point = a+b+c
        return point, key1, key2

    def generate_permutation_key(self, size=14):
        """
        Generate a random permutation of list(range(size))

        Parameters:
        size (int): size of list

        Returns:
        key ([str]): the permuted list in binary form
        dec_key ([int]): the permuted list in decimal form
        """
        key_dec = list(range(size))
        rng = np.random.default_rng()
        rng.shuffle(key_dec)
        f = '0' + str(math.ceil(math.log2(size))) + 'b'
        # get the permutation in binary form
        key = [format(x, f) for x in key_dec]
        return key, key_dec

    def sample_challenge_point(self, point, size=14):
        """
        Sample a random point q from the distribution in which with approx.
        probability 0.5, point (the parameter) is sampled,
        and with approx. probability 0.5, a random point excluding point (the parameter) is uniformly chosen

        Parameters:
        point (str): the point that will be sampled with probability 0.5 in the distribution
        size (int): number of qubits that point encrypts for

        Returns:
        sample (str): challenge point taken from distribution
        """
        # generate a random valid point that has a permutation and one-time pad keys
        key1, key_dec = self.generate_permutation_key()
        key1 = "".join(key1)

        key2_dec = random.randint(0, (2**(size*2))-1)
        key2_bin = format(key2_dec, '0'+str(size*2)+'b')

        random_point = str(key1) + str(key2_bin)
        # keep sampling for a random point uniformly until the random_point is not equivalent to point
        while random_point == point:
            key2_dec = random.randint(0, (2**(size*2))-1)
            key2_bin = format(key2_dec, '0'+str(size*2)+'b')
            random_point = str(key1) + str(key2_bin)
        # sample from challenge distribution in which with approx. 50%, random_point is selected, and 50%, point is selected
        sample = np.random.choice([random_point, point])
        return sample

    def protect(self, permuted_cnots, hadamards, x_key, z_key, init_data=[0, 0, 0, 0, 0, 0, 0, '+', 0, 0, 0, 0, 0, 0], size=14):
        """
        Encodes a program

        Parameters:
        permuted_cnots ([[int,int]]): all permuted CNOT gates to be applied
        hadamards ([int]): all hadamard gates to be applied
        x_key ([int]): all pauli-X gates to be applied
        z_key ([int]): all pauli-Z gates to be applied
        init_data (list): initialized qubit states
        size (int): size of the quantum circuit

        Returns:
        circuit (qiskit's QuantumCircuit): encoded program
        """
        # initialize quantum circuit
        qr = QuantumRegister(size)
        circuit = QuantumCircuit(qr)

        # initialize the states of the quantum circuit
        for i in range(size):
            if init_data[i] == '+':
                circuit.h(i)
            elif init_data[i] == 1:
                circuit.x(i)
            elif init_data[i] == '-':
                circuit.x(i)
                circuit.h(i)
        circuit.barrier()
        # apply delegated one-time pad
        for i in range(size):
            if x_key[i] == 1 and init_data[i] == 0:
                circuit.x(i)
            elif z_key[i] == 1 and init_data[i] == '+':
                circuit.z(i)
        circuit.barrier()
        # apply hadamard gates
        for i in hadamards:
            circuit.h(i)
        circuit.barrier()
        # apply cnot gates
        for cnots in permuted_cnots:
            circuit.cx(cnots[0], cnots[1])
        circuit.barrier()
        return circuit

    def get_syndrome_circuit(self, challenge_input, program, size=14, syndrome_cnots=[[0, 14], [2, 14], [4, 14], [6, 14], [1, 15], [2, 15], [5, 15], [6, 15], [3, 16], [4, 16], [5, 16], [6, 16], [7, 17], [9, 17], [11, 17], [13, 17], [8, 18], [9, 18], [12, 18], [13, 18], [10, 19], [11, 19], [12, 19], [13, 19]]):
        """
        Creates a circuit that detects for single bit and phase flip errors

        Parameters:
        challenge_input (str): point used to decrypt program
        program (qiskit's QuantumCircuit): program for finding error syndromes
        size (int): the number of qubits in the program
        syndrome_cnots ([int,int]): CNOT gates for obtaining error syndromes

        Returns:
        syndrome_circuit (qiskit's QuantumCircuit): program for calculating error syndromes
        """
        key1, key2 = self.point_to_keys(challenge_input)
        # initialize quantum circuit
        qr = QuantumRegister(size+int(size/7*3))
        cr = ClassicalRegister(size+int(size/7*3))
        syndrome_circuit = QuantumCircuit(qr, cr)
        # add program to new quantum circuit
        syndrome_circuit.append(program, range(size))
        # apply gates to decrypt the circuit
        for i in range(size, size+int(size/7*3)):
            syndrome_circuit.h(i)

        for gate in syndrome_cnots:
            syndrome_circuit.cx(gate[1], key1.index(gate[0]))

        for i in range(size, size+int(size/7*3)):
            syndrome_circuit.h(i)

        syndrome_circuit.barrier()
        syndrome_circuit.measure(qr, cr)
        return syndrome_circuit

    def get_syndrome_circuit_mit_measures(self, mit_values, challenge_input, program, size=14, syndrome_cnots=[[0, 14], [2, 14], [4, 14], [6, 14], [1, 15], [2, 15], [5, 15], [6, 15], [3, 16], [4, 16], [5, 16], [6, 16], [7, 17], [9, 17], [11, 17], [13, 17], [8, 18], [9, 18], [12, 18], [13, 18], [10, 19], [11, 19], [12, 19], [13, 19]]):
        """
        Creates a circuit that detects bit and phase flip errors but measures only a subset of qubits;
        Used for tensored error mitigation

        Parameters:
        mit_values ([int]): subset of qubits to be measured
        challenge_input (str): point used to decrypt program
        program (qiskit's QuantumCircuit): program for finding error syndromes
        size (int): the number of qubits in the program
        syndrome_cnots ([int,int]): CNOT gates for obtaining error syndromes

        Returns:
        syndrome_program (qiskit's QuantumCircuit): program for calculating error syndromes with partial qubit measurement
        """
        key1, key2 = self.point_to_keys(challenge_input)

        qr = QuantumRegister(size+int(size/7*3))
        cr = ClassicalRegister(len(mit_values))
        syndrome_program = QuantumCircuit(qr, cr)
        syndrome_program.append(program, range(size))

        for i in range(size, size+int(size/7*3)):
            syndrome_program.h(i)

        for gate in syndrome_cnots:
            syndrome_program.cx(gate[1], key1.index(gate[0]))

        for i in range(size, size+int(size/7*3)):
            syndrome_program.h(i)

        syndrome_program.barrier()
        for i in range(len(mit_values)):
            syndrome_program.measure(qr[mit_values[i]], cr[i])
        return syndrome_program

    def point_to_keys(self, point, size=14):
        """
        Derives the permutation and one-time pad keys from a point

        Parameters:
        point(str): point for deriving keys from
        size (int): number of qubits in program

        Returns:
        circuit (circuit): protected program
        """
        inc = math.ceil(math.log2(size))
        key1 = [int(point[i:i+inc], 2)
                for i in range(0, len(point[:-size*2]), inc)]
        key2_x = [int(value) for value in point[-size*2:-size]]
        key2_z = [int(value) for value in point[-size:]]
        return key1, [key2_x, key2_z]

    def permute_classical(self, key1, orig_cnots, hadamards=[1, 2, 3, 8, 9, 10], size=14):
        """
        Provides the locations of CNOT and Hadamard gates based on a permutated list

        Parameters:
        key1 ([int]): permutated list
        orig_cnots ([[int,int]]): the location of unpermuted CNOT gates
        hadamards ([int]): the location of unpermuted Hadamard gates
        size (int): number of qubits in program

        Returns:
        new_cnot_gates ([[int,int]]): permuted CNOT gates
        new_hadamard_gates ([int]): permuted Hadamard gates
        """
        new_hadamard_gates = [0]*len(hadamards)
        new_cnot_gates = [0]*len(orig_cnots)

        for i in range(len(orig_cnots)):
            new_cnot_gates[i] = [key1.index(
                orig_cnots[i][0]), key1.index(orig_cnots[i][1])]
        for i in range(len(hadamards)):
            new_hadamard_gates[i] = key1.index(hadamards[i])

        return new_cnot_gates, new_hadamard_gates

    def get_OTP_classical_key(self, key, permutation_key, cnots, hadamards):
        """
        Gets the delegated one-time pad key, where the one-time pad key is delegated to the beginning of the program

        Parameters:
        key ([[int],[int]]): the one-time pad key to be delegated
        permutation_key ([int]): permutation
        cnots ([[int,int]]): all CNOT gates
        hadamards ([int]): all Hadamard gates

        Returns:
        new_x_key ([int]): delegated Pauli-X gates of one-time pad
        new_z_key ([int]): delegated Pauli-Z gates of one-time pad
        """
        x_key = key[0]
        z_key = key[1]

        for cnot in cnots:
            a = x_key[cnot[0]]
            b = z_key[cnot[0]]
            c = x_key[cnot[1]]
            d = z_key[cnot[1]]
            x_key[cnot[0]] = a
            z_key[cnot[0]] = b+d
            x_key[cnot[1]] = a+c
            z_key[cnot[1]] = d

        for i in hadamards:
            x_key[i], z_key[i] = z_key[i], x_key[i]

        new_x_key = [i % 2 for i in x_key]
        new_z_key = [i % 2 for i in z_key]

        return new_x_key, new_z_key

    def undo_circuit(self, point, program, rev_cnots=[[3, 6], [3, 5], [3, 4], [2, 6], [2, 4], [2, 0], [1, 5], [1, 4], [1, 0], [0, 6], [0, 5], [10, 13], [10, 12], [10, 11], [9, 13], [9, 11], [9, 7], [8, 12], [8, 11], [8, 7], [7, 13], [7, 12]], size=14):
        """
        Applies all the operations in reverse order as to undo the original program

        Parameters:
        point (str): the point for encoding the program
        program (qiskit's QuantumCircuit): circuit to be undoed
        rev_cnots ([[int,int]]): the reverse sequence of CNOT gates that were applied in the program
        size (int): number of qubits in program

        Returns:
        undo_circuit (qiskit's QuantumCircuit): the program that has been undoed
        """
        key1, key2 = self.point_to_keys(point)
        permuted_cnots, hg = self.permute_classical(key1, rev_cnots)
        qr = QuantumRegister(size)
        cr_trap = ClassicalRegister(size)
        undo_circuit = QuantumCircuit(qr, cr_trap)
        undo_circuit.append(program, range(size))

        for cnot in permuted_cnots:
            undo_circuit.cx(cnot[0], cnot[1])

        undo_circuit.barrier()
        for gate in hg:
            undo_circuit.h(gate)
        undo_circuit.barrier()

        undo_circuit.measure(qr, cr_trap)
        return undo_circuit

    def reverse_cnots(self, cnots):
        """
        Reverse the order of CNOTs

        Parameters:
        cnots ([[int,int]]): original order of cnots

        Returns:
        rev_cnots ([[int,int]]): reversed order of cnots
        """
        rev_cnots = []
        for i in range(len(cnots)):
            rev_cnots.append(cnots[len(cnots)-i-1])
        return rev_cnots

    def get_random_mit_pattern_single(self, size=20, num_qubits=10):
        """
        Selected single qubit pattern for tensored error mitigation

        Parameters:
        size(int): total number of qubits in the program
        num_qubits(int): number of qubits to be selected

        Returns:
        mit_pattern (list): pattern for tensored error mitigation, comprised of single qubits
        mit_values ([int]): a random subset of all qubits in mit_pattern
        """
        mit_vals = random.sample(list(range(size)), num_qubits)
        mit_pattern = [[x] for x in mit_vals]
        return mit_pattern, mit_vals

    def get_permuted_cnots(self, permutation_key, cnots):
        """
        Gets the permuted set of CNOTs to be applied for the syndrome programs

        Parameters:
        permutation_key([int]): permutation
        cnots([[int,int]]): CNOT gates to be applied

        Returns:
        new_permuted_cnots ([[int,int]]): permutation of CNOT gates
        """
        num_aux_qubits = int((len(permutation_key)/7)*3)
        # get the list of auxiliary qubits for obtaining error syndromes
        aux_qubits = list(range(len(permutation_key),
                          len(permutation_key)+num_aux_qubits))
        key = permutation_key + aux_qubits
        new_permuted_cnots = [0]*len(cnots)
        for i in range(len(cnots)):
            new_permuted_cnots[i] = [
                key.index(cnots[i][0]), key.index(cnots[i][1])]
        return new_permuted_cnots

    def get_random_mit_pattern_all(self, permutation_key, steane_cnots=[[1, 0], [1, 4], [1, 5], [2, 0], [2, 4], [2, 6], [3, 4], [3, 5], [3, 6], [7, 12], [7, 13], [8, 7], [8, 11], [8, 12], [9, 7], [9, 11], [9, 13], [10, 11], [10, 12], [10, 13]], syndrome_cnots=[[14, 0], [14, 2], [14, 4], [14, 6], [15, 1], [15, 2], [15, 5], [15, 6], [16, 3], [16, 4], [16, 5], [16, 6], [17, 7], [17, 9], [17, 11], [17, 13], [18, 8], [18, 9], [18, 12], [18, 13], [19, 10], [19, 11], [19, 12], [19, 13]], size=20, num_qubits=10):
        """
        Selected single and double qubit patterns for tensored error mitigation

        Parameters:
        permutation_key([int]): permutation
        steane_cnots ([[int,int]]): all cnot gates for the Steane encoding
        syndrome_cnots ([[int,int]]): all cnot gates for calculating the error syndromes
        size(int): total number of qubits in the program
        num_qubits(int): number of qubits to be selected

        Returns:
        mit_pattern (list): pattern for tensored error mitigation, comprised of single and qubit pairs
        mit_values ([int]): a random subset of all qubits in mit_pattern
        """
        permuted_steane_cnots = self.get_permuted_cnots(
            permutation_key, steane_cnots)
        permuted_syndrome_cnots = self.get_permuted_cnots(
            permutation_key, syndrome_cnots)
        cnots = permuted_steane_cnots + permuted_syndrome_cnots
        # number of qubit pairs to include in pattern
        num_cnots = random.choice(range(10//2))
        count = 0
        cnot_pairs = []
        cnot_values = []
        while count != num_cnots:
            val = random.choice(range(len(cnots)))
            if cnots[val] not in cnot_pairs:
                if cnots[val][0] not in cnot_values and cnots[val][1] not in cnot_values:
                    cnot_pairs.append(cnots[val])
                    cnot_values.append(cnots[val][0])
                    cnot_values.append(cnots[val][1])
                    count = count + 1

        singles = random.sample(set(list(range(20))) -
                                set(cnot_values), num_qubits-(num_cnots*2))
        s = [[x] for x in singles[:]]
        mit_values = cnot_values + singles
        mit_patterns = cnot_pairs + s
        return mit_patterns, mit_values

    def prepare_meas_filter(self, mit_pattern, backend, num_shots, size=20):
        """
        Prepare a tensored error mitigation measurement filter based on specified mit_pattern

        Parameters:
        mit_pattern([int]): pattern used for tensored error mitigation
        backend(qiskit's IBMQBackend): specified backend for preparing measurement filter
        num_shots(int): number of shots for backend
        size(int): number of qubits in program

        Returns:
        meas_filter (qiskit's TensoredMeasFitter.filter): prepared measurement filter
        """
        qr = QuantumRegister(size)
        qulayout = range(size)
        meas_calibs, state_labels = tensored_meas_cal(
            mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
        for circ in meas_calibs:
            print(circ.name)
        job = execute(meas_calibs, backend=backend, shots=num_shots)
        cal_results = job.result()
        meas_fitter = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)
        meas_filter = meas_fitter.filter
        return meas_filter
