from qiskit import *


def get_programs_for_test(server, challenge_input, program, permutation_key, sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, rev_cnots, backend, num_shots):
    """
    Prepares circuits for execution

    Parameters:
    server (Server): Server instance
    challenge_input (str): challenge point for testing programs
    program (qiskit's QuantumCicuit): the encoded program for applying tests
    permutation_key ([int]): permutation ordering
    sp_list ([qiskit's QuantumCircuits]): list of prepared syndrome quantum circuits
    sp_mit_single_list ([qiskit's QuantumCircuits]): list of prepared syndrome quantum circuits with partial measurement and single qubit patterns
    sp_mit_all_list ([qiskit's QuantumCircuits]): list of prepared quantum circuits with syndromes with partial measurement and single and qubit pair patterns
    dp_list ([qiskit's QuantumCircuits]): list of prepared undoed quantum circuits
    meas_filter_singles ([qiskit's TensoredMeasFitter.filter]): list of tensored measurement filters for sp_mit_single_list circuits
    meas_filter_alls ([qiskit's TensoredMeasFitter.filter]): list of tensored measurement filters for sp_mit_all_list circuits
    rev_cnots ([[int,int]]): cnot gates to be applied for undoing the circuit
    backend (qiskit's IBMQBackend): specified backend for preparing measurement filter
    num_shots (int): number of shots for backend

    Returns:
    sp_list ([qiskit's QuantumCircuits]): list of prepared syndrome quantum circuits
    sp_mit_single_list ([qiskit's QuantumCircuits]): list of prepared syndrome quantum circuits with partial measurement and single qubit patterns
    sp_mit_all_list ([qiskit's QuantumCircuits]): list of prepared quantum circuits with syndromes with partial measurement and single and double qubit patterns
    dp_list ([qiskit's QuantumCircuits]): list of prepared undoed quantum circuits
    meas_filter_singles ([qiskit's TensoredMeasFitter.filter]): list of tensored measurement filters for sp_mit_single_list circuits
    meas_filter_alls ([qiskit's TensoredMeasFitter.filter]): list of tensored measurement filters for sp_mit_all_list circuits
    mit_pattern_s ([[int]]): subset of single qubits used in tensored error mitigation, based on the circuits sp_mit_single_list
    mit_pattern_all (list): subset of single and double qubits used in tensored error mitigation, based on the circuits sp_mit_all_list
    """
    syndrome_program = server.get_syndrome_circuit(challenge_input, program)
    mit_pattern_s, mit_val_s = server.get_random_mit_pattern_single()
    mit_pattern_all, mit_val_all = server.get_random_mit_pattern_all(
        permutation_key)
    syndrome_program_mit_single = server.get_syndrome_circuit_mit_measures(
        mit_val_s, challenge_input, program)
    syndrome_program_mit_all = server.get_syndrome_circuit_mit_measures(
        mit_val_all, challenge_input, program)
    decoded_program = server.undo_circuit(
        challenge_input, program, rev_cnots=rev_cnots)
    meas_filter_s = server.prepare_meas_filter(
        mit_pattern_s, backend, num_shots)
    meas_filter_all = server.prepare_meas_filter(
        mit_pattern_all, backend, num_shots)
    sp_list = sp_list + [syndrome_program]
    sp_mit_single_list = sp_mit_single_list + [syndrome_program_mit_single]
    sp_mit_all_list = sp_mit_all_list + [syndrome_program_mit_all]
    dp_list = dp_list + [decoded_program]
    meas_filter_singles = meas_filter_singles + [meas_filter_s]
    meas_filter_alls = meas_filter_alls + [meas_filter_all]
    return sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, mit_pattern_s, mit_pattern_all


def prepare_for_test(server, cnots):
    """
    Prepare inputs for test

    Parameters:
    server (Server): instance of Server for preparing inputs
    cnots ([[int,int]]): cnot gates to be applied

    Returns:
    p (str): point
    k1 ([int]): permutation key
    key2 ([[int],[int]]): one-time pad key
    permuted_cnots([[int,int]]):  cnot gates post permutation
    permuted_hadamards ([int]): hadamard gates post permutation
    x_key ([int]): all delegated pauli-X gates to be applied for one-time pad (key2)
    z_key ([int]):  all delegated pauli-Z gates to be applied for one-time pad (key2)
    data (list): qubits' intial states
    """
    p, k1, k2 = server.generate_point()
    key2 = [k2[0][:], k2[1][:]]
    permuted_cnots, permuted_hadamards = server.permute_classical(k1, cnots)
    rev = server.reverse_cnots(permuted_cnots)
    x_key, z_key = server.get_OTP_classical_key(
        k2, k1, rev, permuted_hadamards)
    data = [0]*14
    data[k1.index(7)] = '+'
    return p, k1, key2, permuted_cnots, permuted_hadamards, x_key, z_key, data


def get_transpiled_circuit_and_depth(circuit_list, backend, init_qubits, opt_level, num_seeds):
    """
    Gets the list of transpiled circuits with the least gate depths based on the random seeds of the specified quantum backend

    Parameters:
    circuit_list ([qiskit's QuantumCircuit]): list of circuits to be transpiled
    backend (qiskit's IBMQBackend): specified quantum computer for transpiling the circuits
    init_qubits ([int]): mapping of virtual to physical qubits
    opt_level (int): the optimization level of the transpiled circuits
    num_seeds (int): the number of random seeds to iterate through

    Returns:
    transpiled_list ([qiskit's QuantumCircuit]): transpiled circuits with the least gate depths
    transpiled_depths ([int]): corresponding gate depths of transpiled_list
    """
    transpiled_list = []
    transpiled_depths = []
    for i in range(len(circuit_list)):
        min_circ = transpile(
            circuit_list[i], backend, initial_layout=init_qubits[i])
        min_depth = min_circ.depth()
        for j in range(num_seeds):
            transpiled_circ = transpile(
                circuit_list[i], backend, initial_layout=init_qubits[i], optimization_level=opt_level)
            depth = transpiled_circ.depth()
            if depth < min_depth:
                min_depth = depth
                min_circ = transpiled_circ
        transpiled_list.append(min_circ)
        transpiled_depths.append(min_circ.depth())
    return transpiled_list, transpiled_depths
