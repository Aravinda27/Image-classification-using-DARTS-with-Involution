import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--Output")
args =  parser.parse_args()
# with open(f'{args.Output}/log.txt', 'r') as input_file:
#     train_acc_lines = []
#     valid_acc_lines = []
#     for line in input_file:
#         if 'train_acc' in line:
#             train_acc_lines.append(float(line.strip().split()[-1]))
#         elif 'valid_acc' in line:
#             valid_acc_lines.append(float(line.strip().split()[-1]))
#     with open(f'{args.Output}/train_acc_lines.txt', 'w') as train_output_file:
#         train_output_file.write(str(train_acc_lines))
#     with open(f'{args.Output}/valid_acc_lines.txt', 'w') as valid_output_file:
#         valid_output_file.write(str(valid_acc_lines))
# Open the input and output files
numbers = []
numbers1 = []
with open(f'{args.Output}/log.txt', 'r') as input_file:
    # Iterate over each line in the input file
    for line in input_file:
        # Check if the line contains the string "train" but not "train_acc"
        if "train" in line and "train_acc" not in line:
            # Split the line into parts using spaces as the separator
            parts = line.split()
            # Extract the 3rd last part and convert it to a float
            value = float(parts[-3])
            # Add the value to a list of numbers
        if "valid" in line and "valid_acc" not in line:
            parts = line.split()
            # Extract the 3rd last part and convert it to a float
            value1 = float(parts[-3])

            numbers.append(value)
            numbers1.append(value1)
            # Write the value to the output file
    with open(f'{args.Output}/train_loss.txt', 'w') as output_file:
        output_file.write(str(numbers))
    with open(f'{args.Output}/valid_loss.txt', 'w') as output_file:
        output_file.write(str(numbers1))
