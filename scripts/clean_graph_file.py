# this file serves to clean large data files or do any preprocessing

import sys
import re

def clean_file_write_line(writer, nums):
    n = len(nums) / 2 * 2
    if n == 0:
        return nums
    for i in range(n):
        writer.write(nums[i])
        if i % 2 == 0:
            writer.write('\t')
        else:
            writer.write('\n')
    return nums[n:]

# for some graph files, each line may contain several vertex ids 
# or some strange strings that cannot be converted into vertex ids.
def clean_file(filename):
    write_filename = filename + '.clean'
    fail_write_filename = filename + ".fail"
    writer = open(write_filename, "w")
    fail_writer = open(fail_write_filename, "w")

    n = 0
    fail = 0
    nums = []
    fail_nums = []
    with open(filename, "r") as reader:
        for line in reader:
            if len(line) > 0 and line[0] == '#':
                continue
            strs = re.split("[ ,\t,\n]*", line)
            for s in strs:
                try:
                    a = int(s)
                    nums.append(s)

                    assert s.isdigit()
                    n += 1
                except ValueError:
                    fail_nums.append(s)

                    fail += 1
                    assert s.isdigit() == False
            nums = clean_file_write_line(writer, nums)
            fail_nums = clean_file_write_line(fail_writer, fail_nums)
            
    nums = clean_file_write_line(writer, nums)
    fail_nums = clean_file_write_line(fail_writer, fail_nums)

    assert len(nums) == 0
       
    writer.close()
    fail_writer.close()

    print n
    print fail

def remove_edges(filename, edges_count_limit):
    write_filename = filename + ".limit"
    writer = open(write_filename, "w")
    n = 0

    with open(filename, "r") as reader:
        for line in reader:
            if (len(line)) > 0 and line[0] == '#':
                continue
            writer.write(line)
            n += 1
            if n >= edges_count_limit:
                break
    writer.close()
    print n

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "ARGUMENTS: data_file_name"
        sys.exit()
    #clean_file(sys.argv[1])
    remove_edges(sys.argv[1], 2147473647)
    #remove_edges(sys.argv[1], 10000)

