# processing a single measurement result

import ast
from ProcessingFunctions import *

# set location for retrieving all the measurement outcome results and information
info_file = "./info.csv"
# set location for saving all the individual calculated error rates (i.e. bit, phase, and both bit and phase combined errors)
save_file = "./individual_error_rates.csv"
# set index of single job measurement information in info_file
index = 10
# set a single job's measurement counts
raw = {}

df = pd.read_csv(info_file)
all_key1 = df.challenge_key_1.to_list()
all_key2 = df.challenge_key_2.to_list()
is_point = df.is_point.to_list()

first_steane_codewords = ['0000000','1010101','0110011','1100110','0001111','1011010','0111100','1101001']
second_steane_codewords = ['0000000', '1110000', '1001100', '0111100', '0101010', '1011010', '1100110', '0010110', '1101001', '0011001', '0100101', '1010101', '1000011', '0110011', '0001111', '1111111']
# the codewords of our Steane encoded program
codeword_combos = [x+y for x in first_steane_codewords for y in second_steane_codewords]

counts = reverse_counts(raw)
key1 = ast.literal_eval(all_key1[index])
key2 = ast.literal_eval(all_key2[index])
xkey = key2[0] + [0]*6
zkey = key2[1] + [0]*6

del_x_key, del_z_key = get_delegated_OTP_keys(key1, xkey, zkey)
processed_counts = apply_OTP_and_unpermute(counts, key1, del_x_key, del_z_key)
bit, phase, all = check_correctness(processed_counts,codeword_combos)

print(index, is_point[index], bit, phase, all)

# uncomment for adding error rates to a dataframe
# fields = ['#', 'is_p','no_bit_flip_percentage', 'no_phase_flip_percentage', 'no_error_percentage']
# stats = pd.DataFrame(columns=fields)
# stats.loc[index] = [index, is_point[index], bit, phase, all]
# print(stats)


