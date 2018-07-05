#########################################################
##
## Check difference between given text files
##
#########################################################

import numpy as np
import os
import sys

def get_diff(base, compare):
    base_file = np.loadtxt(base, comments={'Z'})
    compare_file = np.loadtxt(compare, comments={'Z'})

    if base_file.shape != compare_file.shape:
        print ("The two files have differing shape.")
        return 1;

    abs_tolerance = 1e-14
    diff = base_file - compare_file
    thresholded_diff = np.where(diff > abs_tolerance, diff, 0)
    if np.count_nonzero(thresholded_diff) is 0:
        return 0

    relative_tolerance = 1e-6
    rel_error = diff / base_file
    thresholded_rel_error = np.where(rel_error > relative_tolerance, rel_error, 0)
    if np.count_nonzero(thresholded_rel_error) is 0:
        return 0

    return 1

if (len(sys.argv) != 2):
    sys.exit(1)

directory = os.getcwd() + sys.argv[1]
print (directory)
data_dir = directory + "/Data/"
target_dir = directory + "/target/"

data_files = [file for file in os.listdir(data_dir) if file.endswith(".plt")]
target_files = [file for file in os.listdir(data_dir) if file.endswith(".plt")]

if (set(data_files) != set(target_files)):
    print("Files in ./data and ./target are different.")
    sys.exit(1)

catch_error = 0;
for file_name in data_files:
    data_file = data_dir + file_name
    target_file = target_dir + file_name
    difference = get_diff(target_file, data_file)
    if difference is 1:
        print("Files %s are different." % file_name)
        catch_error += 1
if (catch_error != 0):
    sys.exit(1)
else:
    sys.exit(0)




