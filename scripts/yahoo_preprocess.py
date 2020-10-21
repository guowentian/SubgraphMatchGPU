import sys
import random
import re
from sets import Set

def get_int_list(nums):
    ret = []
    for n in nums:
        if n.isdigit():
            ret.append(n)
    return ret


def read_file(read_filename, writer):
    with open(read_filename, "r") as reader:
        for line in reader:
            strs = re.split("\s|\t|\n", line)
            nums = get_int_list(strs)
            if len(nums) == 0: 
                continue
            assert len(nums) >= 2
            #print nums

            n = int(nums[1])
            assert len(nums) == n+2
            for i in range(n):
                writer.write(nums[0] + "\t" +nums[i+2] + "\n")

def extract_yahoo_dataset(input_filename1, input_filename2, output_filename):
    writer = open(output_filename, "w")
    read_file(input_filename1, writer)
    read_file(input_filename2, writer)
    writer.close()

def relabel_node_ids(filename):
    d = {}
    n = 0
    with open(filename, 'r') as reader:
        for line in reader:
            strs = re.split("\s|\t|\n", line)
            nums = get_int_list(strs)
            if len(nums) > 0:
                assert len(nums) == 2
                for i in range(2):
                    if d.get(nums[i]) == None:
                        d[nums[i]] = n
                        n += 1
    return d

def sample_node_ids(n, ratio):
    ret = Set([])
    for v in range(n):
        if random.uniform(0,1) < ratio:
            ret.add(v)
    return ret

def sample_graph(input_filename, output_filename, d, ratio):
    nodes = sample_node_ids(len(d), ratio)
    #print nodes

    writer = open(output_filename, "w")
    with open(input_filename, 'r') as reader:
        for line in reader:
            strs = re.split("\s|\t|\n", line)
            nums = get_int_list(strs)
            if len(nums) > 0:
                assert len(nums) == 2
                v1 = d.get(nums[0])
                v2 = d.get(nums[1])
                if (v1 in nodes) and (v2 in nodes):
                    writer.write(line)
    writer.close()
 
def sample_graphs(input_filename, output_filenames, ratios):
    d = relabel_node_ids(input_filename)
    assert len(output_filenames) == len(ratios)
    for i in range(len(ratios)):
        sample_graph(input_filename, output_filenames[i], d, ratios[i])

if __name__ == "__main__":
    #dataset_dir = "/temp/wentian/datasets/yahoo"
    #extract_yahoo_dataset(dataset_dir + '/ydata-yaltavista-webmap-v1_0_links-1.txt', dataset_dir+ '/ydata-yaltavista-webmap-v1_0_links-2.txt', dataset_dir+'/yahoog2.txt')

    #dataset_dir = "/temp/wentian/datasets/yahoo/test"
    #extract_yahoo_dataset(dataset_dir + '/input1.txt', dataset_dir + '/input2.txt', dataset_dir+'output.txt')

    dataset_dir = "/temp/wentian/datasets/yahoo"
    sample_graphs(dataset_dir+'/yahoog2.txt', [dataset_dir+'/yahoog2_1.txt'], [0.01])

    #dataset_dir = "/temp/wentian/datasets/yahoo/test"
    #sample_graphs(dataset_dir+'/full_graph.txt', [dataset_dir+'/sample_graph_20.txt', dataset_dir+'/sample_graph_60.txt'], [0.2,0.6])

    
