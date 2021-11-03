from qiskit.test.mock import FakeSydney, FakeMontreal
import time
import pandas as pd
from Server import Server
from TestingFunctions import *
import numpy as np
import math

"""Correctness Tests for Password Authentication Scheme"""

# CNOT gates for applying the Steane Code
CNOT_GATES = [[1, 0], [1, 4], [1, 5], [2, 0], [2, 4], [2, 6], [3, 4], [3, 5], [3, 6], [7, 12], [
    7, 13], [8, 7], [8, 11], [8, 12], [9, 7], [9, 11], [9, 13], [10, 11], [10, 12], [10, 13]]
REV_CNOT_GATES = [[10, 13], [10, 12], [10, 11], [9, 13], [9, 11], [9, 7], [8, 12], [8, 11], [
    8, 7], [7, 13], [7, 12], [3, 6], [3, 5], [3, 4], [2, 6], [2, 4], [2, 0], [1, 5], [1, 4], [1, 0]]

"""User Defined Values"""

# COMPUTATION CONFIGURATIONS
NUM_SHOTS = 8192
NUM_DIFF_PROGRAMS = 10
NUM_ITERATIONS = 1
NUM_RANDOM_ITERATIONS = 10

# SPECIFY NUMBER OF RANDOM SEEDS FOR TRANSPILING QUANTUM CIRCUITS AND OPTIMIZATION LEVEL
NUM_SEEDS = 150
OPT_LEVEL = 2

# SET PHYSICAL TO VIRTUAL QUBIT MAPPING OF QUANTUM MACHINE
unpermuted_layout = [8, 11, 13, 19, 14, 20, 16, 1, 2, 7, 12, 6, 4, 10]
syndrome_layout = [5, 9, 18, 0, 3, 15]

# SET FILENAMES FOR DATA SAVING
filename_0 = "./general_info_20Q.txt"
filename_error = "./error_results_20Q.csv"
filename_mit = "./mit_results_20Q.csv"
filename_decoded = "./decoded_results_20Q.csv"

# SET QUANTUM COMPUTER BACKEND OF YOUR CHOICE
fake_mtrl = FakeMontreal()
# BACKEND = AerSimulator.from_backend(fake_mtrl)
BACKEND = Aer.get_backend('qasm_simulator')

server = Server()
start = time.time()
fields = ['is_point', 'point_value', 'challenge_point_value', 'key_1', 'key_2',
          'challenge_key_1', 'challenge_key_2', 'mit_pattern_single', 'mit_pattern_all']

results_info = pd.DataFrame(columns=fields)
results_info_decoded = pd.DataFrame(columns=fields)
sp_list = []
sp_mit_single_list = []
sp_mit_all_list = []
dp_list = []
meas_filter_singles = []
meas_filter_alls = []

"""
Test 1.1: Point = Challenge Input Correctness Check
"""

print("_____________PART A: Challenge Input == Point_____________")
for i in range(NUM_DIFF_PROGRAMS):
    p, k1, key2, cnots, hadamards, x_key, z_key, data = prepare_for_test(
        server, CNOT_GATES)
    program = server.protect(cnots, hadamards, x_key, z_key, init_data=data)
    # set challenge_input
    challenge_input = p
    challenge_key1, challenge_key2 = server.point_to_keys(challenge_input)
    for k in range(NUM_ITERATIONS):
        sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, mit_pattern_s, mit_pattern_all = get_programs_for_test(
            server, challenge_input, program, k1, sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, REV_CNOT_GATES, BACKEND, NUM_SHOTS)
        index = i*NUM_ITERATIONS+k
        results_info.loc[index] = [str(p) == str(challenge_input), p, challenge_input,
                                   k1, key2, challenge_key1, challenge_key2, mit_pattern_s, mit_pattern_all]
        results_info_decoded.loc[index] = [str(p) == str(
            challenge_input), p, challenge_input, k1, key2, challenge_key1, challenge_key2, "-", "-"]


"""
Test 1.2: Point != Challenge Input, w/ 1 permutation error Correctness Check
"""

print("\n_____________PART B: Challenge Input != Point - one Permutation Error_____________")
for j in range(NUM_DIFF_PROGRAMS):
    p, k1, key2, cnots, hadamards, x_key, z_key, data = prepare_for_test(
        server, CNOT_GATES)
    program = server.protect(cnots, hadamards, x_key, z_key, init_data=data)
    # prepare challenge input
    i = np.random.choice(14, 2, False)
    edited_k1 = k1[:]
    edited_k1[i[0]], edited_k1[i[1]] = k1[i[1]], k1[i[0]]
    f = '0' + str(math.ceil(math.log2(14))) + 'b'
    new_key1 = [format(x, f) for x in edited_k1]
    challenge_input = str("".join(new_key1)) + str(p[-28:])
    challenge_key1, challenge_key2 = server.point_to_keys(challenge_input)

    for k in range(NUM_ITERATIONS):
        sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, mit_pattern_s, mit_pattern_all = get_programs_for_test(
            server, challenge_input, program, k1, sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, REV_CNOT_GATES, BACKEND, NUM_SHOTS)
        index = j*NUM_ITERATIONS+k + (NUM_ITERATIONS*NUM_DIFF_PROGRAMS)
        results_info_decoded.loc[index] = [str(p) == str(
            challenge_input), p, challenge_input, k1, key2, challenge_key1, challenge_key2, "-", "-"]
        results_info.loc[index] = [str(p) == str(challenge_input), p, challenge_input,
                                   k1, key2, challenge_key1, challenge_key2, mit_pattern_s, mit_pattern_all]

"""
Test 1.3: Point != Challenge Input, w/ 1 X error Correctness Check
"""

print("\n_____________PART C: Challenge Input != Point - one X Error_____________")
for j in range(NUM_DIFF_PROGRAMS):
    p, k1, key2, cnots, hadamards, x_key, z_key, data = prepare_for_test(
        server, CNOT_GATES)
    program = server.protect(cnots, hadamards, x_key, z_key, init_data=data)
    # prepare challenge input
    i = np.random.choice(14, 1, False)
    index = (i[0]-28)
    challenge_input = str(p[:index]) + \
        str((int(p[index])+1) % 2) + str(p[index+1:])
    challenge_key1, challenge_key2 = server.point_to_keys(challenge_input)
    for k in range(NUM_ITERATIONS):
        sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, mit_pattern_s, mit_pattern_all = get_programs_for_test(
            server, challenge_input, program, k1, sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, REV_CNOT_GATES, BACKEND, NUM_SHOTS)
        index = j*NUM_ITERATIONS+k + (NUM_ITERATIONS*NUM_DIFF_PROGRAMS*2)
        results_info_decoded.loc[index] = [str(p) == str(
            challenge_input), p, challenge_input, k1, key2, challenge_key1, challenge_key2, "-", "-"]
        results_info.loc[index] = [str(p) == str(challenge_input), p, challenge_input,
                                   k1, key2, challenge_key1, challenge_key2,  mit_pattern_s, mit_pattern_all]
print(len(results_info))

"""
Test 1.4: Point != Challenge Input, w/ 1 Z error Correctness Check
"""

print("\n_____________PART D: Challenge Input != Point - one Z-Error_____________")
for j in range(NUM_DIFF_PROGRAMS):
    p, k1, key2, cnots, hadamards, x_key, z_key, data = prepare_for_test(
        server, CNOT_GATES)
    program = server.protect(cnots, hadamards, x_key, z_key, init_data=data)
    # prepare challenge input
    i = np.random.choice(14, 1, False)
    index = (i[0]-14)
    print(index)
    if i == 13:
        challenge_input = str(p[:index]) + str((int(p[index])+1) % 2)
    else:
        challenge_input = str(p[:index]) + \
            str((int(p[index])+1) % 2) + str(p[index+1:])
    challenge_key1, challenge_key2 = server.point_to_keys(challenge_input)

    for k in range(NUM_ITERATIONS):
        sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, mit_pattern_s, mit_pattern_all = get_programs_for_test(
            server, challenge_input, program, k1, sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, REV_CNOT_GATES, BACKEND, NUM_SHOTS)
        index = j*NUM_ITERATIONS+k + (NUM_ITERATIONS*NUM_DIFF_PROGRAMS*3)
        results_info.loc[index] = [str(p) == str(challenge_input), p, challenge_input,
                                   k1, key2, challenge_key1, challenge_key2, mit_pattern_s, mit_pattern_all]
        results_info_decoded.loc[index] = [str(p) == str(
            challenge_input), p, challenge_input, k1, key2, challenge_key1, challenge_key2,  "-", "-"]

"""
Test 1.5: Point != Challenge Input, w/ 1 X, Z error EACH
"""

print("\n_____________PART E: Challenge Input != Point - one X and Z Error_____________")
for j in range(NUM_DIFF_PROGRAMS):
    p, k1, key2, cnots, hadamards, x_key, z_key, data = prepare_for_test(
        server, CNOT_GATES)
    program = server.protect(cnots, hadamards, x_key, z_key, init_data=data)
    # prepare challenge input
    i = np.sort(np.random.choice(14, 2, True))
    x_error_index = i[0] - 28
    z_error_index = i[1] - 14
    if i[1] == 13:
        challenge_input = str(p[:x_error_index]) + str((int(p[x_error_index])+1) %
                                                       2) + str(p[x_error_index+1: z_error_index])
    else:
        challenge_input = str(p[:x_error_index]) + str((int(p[x_error_index])+1) % 2) + str(
            p[x_error_index+1: z_error_index]) + str((int(p[z_error_index])+1) % 2) + str(p[z_error_index+1:])
    challenge_key1, challenge_key2 = server.point_to_keys(challenge_input)

    for k in range(NUM_ITERATIONS):
        sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, mit_pattern_s, mit_pattern_all = get_programs_for_test(
            server, challenge_input, program, k1, sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, REV_CNOT_GATES, BACKEND, NUM_SHOTS)
        index = j*NUM_ITERATIONS+k + (NUM_ITERATIONS*NUM_DIFF_PROGRAMS*4)
        results_info.loc[index] = [str(p) == str(challenge_input), p, challenge_input,
                                   k1, key2, challenge_key1, challenge_key2, mit_pattern_s, mit_pattern_all]
        results_info_decoded.loc[index] = [str(p) == str(
            challenge_input), p, challenge_input, k1, key2, challenge_key1, challenge_key2, "-", "-"]

"""
Test 1.6: Point != Challenge Input, w/ random error Correctness Check
"""

print("\n_____________PART F: Random Challenge Input_____________")
for i in range(NUM_RANDOM_ITERATIONS):
    p, k1, key2, cnots, hadamards, x_key, z_key, data = prepare_for_test(
        server, CNOT_GATES)
    program = server.protect(cnots, hadamards, x_key, z_key, init_data=data)
    # prepare challenge input
    challenge_input = server.sample_challenge_point(p)
    challenge_key1, challenge_key2 = server.point_to_keys(challenge_input)

    sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, mit_pattern_s, mit_pattern_all = get_programs_for_test(
        server, challenge_input, program, k1, sp_list, sp_mit_single_list, sp_mit_all_list, dp_list, meas_filter_singles, meas_filter_alls, REV_CNOT_GATES, BACKEND, NUM_SHOTS)
    index = (NUM_ITERATIONS*NUM_DIFF_PROGRAMS*5)+i
    results_info.loc[index] = [str(p) == str(challenge_input), p, challenge_input,
                               k1, key2, challenge_key1, challenge_key2, mit_pattern_s, mit_pattern_all]
    results_info_decoded.loc[index] = [str(p) == str(
        challenge_input), p, challenge_input, k1, key2, challenge_key1, challenge_key2,  "-", "-"]

"""
Execute Circuits prepared in Tests
"""

# getting the virtual to physical qubit mapping for all circuits
# mapping is based on the permutation of the circuits and the ideal physical ordering of the quantum computer
init_qubits = []
init_qubits_msg = []
for key1 in results_info.challenge_key_1:
    k1 = key1[:]
    for i in range(len(k1)):
        k1[i] = unpermuted_layout[k1[i]]
    init_qubits_msg.append(k1[:])
    for j in syndrome_layout:
        k1.append(j)
    init_qubits.append(k1)

# getting all the transpiled circuits
transpiled_sp_list, transpiled_sp_depths = get_transpiled_circuit_and_depth(
    sp_list, BACKEND, init_qubits, OPT_LEVEL, NUM_SEEDS)
transpiled_sp_singles_list, transpiled_sp_singles_depths = get_transpiled_circuit_and_depth(
    sp_mit_single_list, BACKEND, init_qubits, OPT_LEVEL, NUM_SEEDS)
transpiled_sp_all_list, transpiled_sp_all_depths = get_transpiled_circuit_and_depth(
    sp_mit_all_list, BACKEND, init_qubits, OPT_LEVEL, NUM_SEEDS)
transpiled_sp_msg_list, transpiled_sp_msg_depths = get_transpiled_circuit_and_depth(
    dp_list, BACKEND, init_qubits_msg, OPT_LEVEL, NUM_SEEDS)

"""> ### Run Transpiled Circuits on Quantum Computers & Saving to Files"""

# execute jobs of transpiled error syndrome programs
job = execute(transpiled_sp_list, BACKEND, shots=NUM_SHOTS)
results_sim = job.result()
counts = results_sim.get_counts()
counts = [str(x) for x in counts]

# saving data
results_info.insert(9, "device_counts", counts)
results_info.insert(10, "circuit_depth", transpiled_sp_depths)
results_info.to_csv(filename_error)

mit_counts_singles = []
mit_counts_all = []

# execute jobs of transpiled error syndrome programs (with partial qubit measurement) for error mitigation
job_s = execute(transpiled_sp_singles_list, BACKEND, shots=NUM_SHOTS)
job_all = execute(transpiled_sp_all_list, BACKEND, shots=NUM_SHOTS)
results_sim_s = job_s.result()
counts_s = results_sim_s.get_counts()
results_sim_all = job_all.result()
counts_all = results_sim_all.get_counts()

# get the mitigated counts of the transpiled error syndrome (with partial qubit measurement)
for j in range(NUM_ITERATIONS*NUM_DIFF_PROGRAMS*5 + NUM_RANDOM_ITERATIONS):
    mitigated_counts = meas_filter_singles[j].apply(counts_s[j])
    mit_counts_singles.append(str(mitigated_counts))

    mitigated_counts = meas_filter_alls[j].apply(counts_all[j])
    mit_counts_all.append(str(mitigated_counts))

# saving data
results_info.insert(11, "raw_singles", counts_s)
results_info.insert(12, "raw_all", counts_all)
results_info.insert(13, "mitigated_counts_singles", mit_counts_singles)
results_info.insert(14, "singles_circuit_depth", transpiled_sp_singles_depths)
results_info.insert(15, "mitigated_counts_all", mit_counts_all)
results_info.insert(16, "all_circuit_depth", transpiled_sp_all_depths)
results_info.to_csv(filename_mit)

# execute jobs of undoed programs
job = execute(transpiled_sp_msg_list, BACKEND, shots=NUM_SHOTS)
results_sim = job.result()
de_counts = results_sim.get_counts()

# saving data
results_info_decoded.insert(8, "device_counts", de_counts)
results_info_decoded.insert(9, "circuit_depth", transpiled_sp_msg_depths)
results_info_decoded.to_csv(filename_decoded)

# saving some generic data
with open(filename_0, 'w') as writefile:
    x = time.time() - start
    writefile.write("--------------------ELAPSED TIME: \n")
    writefile.write(str(x))
    writefile.write(
        "\n________________________________COUNTS_____________________________________\n")
    writefile.write(str(counts))
    writefile.write(
        "\n________________________________DECODED_COUNTS_____________________________________\n")
    writefile.write(str(de_counts))
