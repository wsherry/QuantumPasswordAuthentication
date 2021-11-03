# processing all measurement results from a file

import ast
from ProcessingFunctions import *
import pandas as pd

# set location of data file containing a list of raw counts only
# format of file: "["{'00000000000000000000':8192}"]"
data = "raw_counts_data.txt"

info_file = "./info.csv"
# set location for saving all the individual calculated error rates (i.e. bit, phase, and both bit and phase combined errors)
save_file = "./individual_error_rates.csv"

df = pd.read_csv(info_file)
all_key1 = df.challenge_key_1.to_list()
all_key2 = df.challenge_key_2.to_list()
is_point = df.is_point.to_list()

fields = ['#', 'is_p','no_bit_flip_percentage', 'no_phase_flip_percentage', 'no_error_percentage']
stats = pd.DataFrame(columns=fields)

# all the steane codewords from encoded program
first_steane_codewords = ['0000000','1010101','0110011','1100110','0001111','1011010','0111100','1101001']
second_steane_codewords = ['0000000', '1110000', '1001100', '0111100', '0101010', '1011010', '1100110', '0010110', '1101001', '0011001', '0100101', '1010101', '1000011', '0110011', '0001111', '1111111']
codeword_combos = [x+y for x in first_steane_codewords for y in second_steane_codewords]

raw_data = ""
with open(data) as f:
  raw_data = f.read()
raw_data = ast.literal_eval(raw_data)

index = 0
for x in raw_data:
  raw = ast.literal_eval(x)
  counts = reverse_counts(raw)
  key1 = ast.literal_eval(all_key1[index])
  key2 = ast.literal_eval(all_key2[index])
  xkey = key2[0] + [0]*6
  zkey = key2[1] + [0]*6
  x_key, z_key = get_delegated_OTP_keys(key1, xkey, zkey)
  processed_counts = apply_OTP_and_unpermute(counts, key1, x_key, z_key)
  bit, phase ,all = check_correctness(processed_counts, codeword_combos)
  print(is_point[index], bit, phase, all)
  stats.loc[index] = [index, is_point[index], bit, phase, all]
  index = index +1

stats.to_csv(save_file)

print(stats)

df = get_average_rates(save_file, num_tests = 5, num_iterations= 10)
print(df)

random_df = get_average_rates_from_random_tests(save_file, 50, 60)
print(random_df)