from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reverse_counts(counts, size=20):
  """ 
  Reverses the keys of a dictionary (i.e. the characters in all the keys are reversed)

  Parameters:
  counts (dict): dictionary containing the measurement results
  size (int): the number of qubits measured

  Returns:
  reverse_counts (dict): dictionary with keys in reverse order
  """
  intermediate = {}
  for key, value in counts.items():
    rev_key = ""
    for i in range(size):
      rev_key = rev_key + key[size-i-1]
    intermediate[key] = rev_key
  reverse_counts = dict([(intermediate.get(k), v) for k, v in counts.items()])
  return reverse_counts

def get_delegated_OTP_keys(permutation, x_key, z_key, num_qubits=14, syndrome_cnots = [[14, 0], [14, 2], [14, 4], [14, 6], [15, 1], [15, 2], [15, 5], [15, 6], [16, 3], [16, 4], [16, 5], [16, 6], [17, 7], [17, 9], [17, 11], [17, 13], [18, 8], [18, 9], [18, 12], [18, 13], [19, 10], [19, 11], [19, 12], [19, 13]]):
  """ 
  Get delegated, post-processed, classical one-time pad keys for a program

  Parameters:
  permutation ([int]): permutation key
  x_key ([int]): X part of the non-delegated one-time pad key
  z_key ([int]): Z part of the non-delegated one-time pad key
  num_qubits (int): number of data qubits
  syndrome_cnots ([[int,int]]): all cnot gates used to derive error syndromes

  Returns:
  delegated_x_key ([int]): classically processed and delegated X part of one-time pad key
  delegated_z_key ([int]): classically processed and delegated Z part of one-time pad key
  """
  permuted_cnots = []
  for gate in syndrome_cnots:
    permuted_cnots.append([gate[0],permutation.index(gate[1])])

  new_x_key = x_key[:]
  new_z_key = z_key[:]

  for cnot in permuted_cnots:
    a = new_x_key[cnot[0]]
    b = new_z_key[cnot[0]]
    c = new_x_key[cnot[1]]
    d = new_z_key[cnot[1]]
    new_x_key[cnot[0]] = a
    new_z_key[cnot[0]] = b+d
    new_x_key[cnot[1]] = a+c
    new_z_key[cnot[1]] = d

  #hadamard operator delegation
  for i in range(num_qubits,num_qubits + int(num_qubits/7*3)):
    new_x_key[i], new_z_key[i] =  new_z_key[i], new_x_key[i]

  delegated_x_key = [i%2 for i in new_x_key]
  delegated_z_key = [i%2 for i in new_z_key]
  return delegated_x_key, delegated_z_key

def apply_OTP_and_unpermute(counts, permutation, x_key, z_key, num_qubits=14):
  """
  Classical processing of quantum measurement outcomes
  Includes applying the delegated one-time pad and unpermuting the circuit

  Parameters:
  counts (dict): all the measurement outcomes for a job
  permutation([int]): permutation key
  x_key ([int]): x gates part of one-time pad key
  z_key ([int]): z gates part of one-time pad key 
  num_qubits (int): number of data qubits

  Returns:
  unpermuted_steane(dict): classically post processed measurement outcomes
  """
  processed_results = {}
  for key, value in counts.items():
    new_key = ""
    for i in range(num_qubits + int(num_qubits/7*3)):
      val = int(key[i])
      k2_val = int(x_key[i])
      if k2_val == 1 and val == 0:
        new_key = new_key + "1"
      elif k2_val == 1 and val == 1:
        new_key = new_key + "0"
      else:
        new_key = new_key + str(val)
    processed_results[new_key] = value

  unpermuted_steane = {}
  for key, value in processed_results.items():
    new_key = ""
    for i in range(num_qubits):
      new_key = new_key+ key[permutation.index(i)]
    syndrome_keys=""
    for j in range(int(num_qubits/7*3)):
      syndrome_keys = syndrome_keys + key[-int(int(num_qubits/7*3)-j)]
    new_key = new_key + syndrome_keys
    # print(syndrome_keys)
    # print(new_key)
    unpermuted_steane[new_key] = value
  return unpermuted_steane

def check_correctness(counts, codeword_combos, syndrome = '000000', num_shots = 8192, num_qubits = 14):
  """ 
  Gets the correct measurement outcome rates of a job

  Parameters:
  counts (dict): all processed measurement outcomes
  codeword_combos ([str]): all codewords
  syndrome (str): the correct no error syndrome
  num_shots (int): the number of times the computation was run
  num_qubits (int): the number of data qubits

  Returns:
  bit_rate (float): rate of measurement outcomes that have no bit flips (i.e. no bit error)
  phase_rate (float): rate of measurement outcomes that have no phase flips (i.e. no phase error)
  all_rate (float): rate of measurement outcomes that have no bit or phase flips (i.e. no bit and phase error)
  """

  bit_count = 0
  phase_count = 0
  all_count = 0
  for key, val in counts.items():
    if key[:num_qubits] in codeword_combos:
      bit_count = bit_count + 1
      if key[num_qubits:] == syndrome:
        all_count = all_count +1
    if key[num_qubits:] == syndrome:
      phase_count = phase_count +1
  bit_rate = bit_count/num_shots
  phase_rate = phase_count/num_shots
  all_rate = all_count/num_shots
  return bit_rate, phase_rate, all_rate

def get_average_rates(file_name, num_tests = 5, num_iterations= 10):
  """ 
  Gets the average true positive and false positive rates for the different tests
  For tests where the challenge input is equal to the password, the average true positive rate is found.
  In all other cases, the average false positive is found.
  
  Parameters:
  file_name (str): the name of the file in which the rates for the individual rates were saved
  num_tests (int): the number of different tests performed
  num_iterations (int): the number of iterations each test was performed

  Returns:
  new_df (panda's DataFrame): contains the averages of all the tests

  """
  try:
    df = pd.read_csv(file_name)
  except error as err:
    print("Error: ", err)
  
  new_df = pd.DataFrame()

  for i in range(num_tests):
    avgs = df[i*num_iterations:(i+1)*num_iterations].mean()
    new_df[str(i)] = avgs
  return new_df

def get_average_rates_from_random_tests(file_name, start_index, end_index):
  """ 
  Gets the average true positive and false positive rates for tests that sample random challenge inputs
  For tests where the challenge input is equal to the password, the average true positive rate is found.
  In all other cases, the average false positive is found.
  
  Parameters:
  file_name (str): the name of the file in which the rates for the individual rates were saved
  start_index (int): the location of where random tests starts according to data ordered in file_name
  end_index (int): the location of where random tests ends according to data ordered in file_name

  Returns:
  new_df (panda's DataFrame): contains the averages of the random tests

  """
  try:
    df = pd.read_csv(file_name)
  except Error as err:
    print("Error: ", err)
  new_df = pd.DataFrame()

  random_avgs = df[start_index:end_index].groupby(['is_p']).get_group(True).mean()
  new_df["True Positive"] = random_avgs
  random_avgs = df[start_index:end_index].groupby(['is_p']).get_group(False).mean()
  new_df["False Positive"] = random_avgs
  return new_df