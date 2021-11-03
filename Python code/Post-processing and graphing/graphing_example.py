import collections
import matplotlib.pyplot as plt

# set no error phase syndrome
phase_syndrome = '000000'
# set the post-processed measurement outcomes
counts_dict = {}
# set the number of data qubits used 
num_qubits = 14
# set the number of syndrome qubits used
num_syndrome = 6

# all the steane codewords from encoded program
first_steane_codewords = ['0000000','1010101','0110011','1100110','0001111','1011010','0111100','1101001']
second_steane_codewords = ['0000000', '1110000', '1001100', '0111100', '0101010', '1011010', '1100110', '0010110', '1101001', '0011001', '0100101', '1010101', '1000011', '0110011', '0001111', '1111111']
codeword_combos = [x+y for x in first_steane_codewords for y in second_steane_codewords]

d = collections.OrderedDict(sorted(counts_dict.items()))
count = 0
# set color of all the wrong measurement outcomes
colors = ['lightgray']*len(d)
patterns = ['']*len(d)
for key, val in d.items():
  if phase_syndrome == key[-num_syndrome:]:
    if key[:num_qubits] in codeword_combos:
      # set color of all the right measurement outcomes
      colors[count]= "black"
  count = count +1 
x_vals = list(d.keys())
y_vals = list(d.values())

plt.figure(figsize=(20,14))
for i in range(len(d)):
  plt.bar(x_vals[i], y_vals[i], color=colors[i])
plt.xticks(fontsize=18, rotation=90)
plt.yticks(fontsize=18)
plt.xlabel('Measurement Values', fontsize=25)
plt.ylabel('Probability', fontsize=25)
plt.title('Quantum Computer without Err Mit', fontsize=30)
plt.show()