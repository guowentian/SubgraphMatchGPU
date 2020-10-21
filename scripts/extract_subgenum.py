from extract_base import extract_keywords_time
from extract_base import extract_keywords_multi_time
from extract_base import extract_keywords_count

import csv


def profile_phases(datasets=None, queries=None, methods=None, feature=None, keywords=None, input_dir='reuse'):
    print keywords

    for dataset in datasets:
        for method in methods:
            print "dataset=%s,method=%d" % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + "_q" + str(query) + "_" + feature + ".txt"
                times = []
                for keyword in keywords:
                    t = extract_keywords_time(filename, keyword)
                    times.append(t)
                ss = ''
                for i in range(len(times)):
                    if i > 0:
                        ss += ' '
                    ss += times[i]
                print ss


def extract_perf(datasets=None, queries=None, methods=None, feature=None, input_dir='logs'):
    for dataset in datasets:
        for method in methods:
            print 'dataset=%s,method=%d' % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + \
                    '_a' + str(method) + "_q" + str(query)
                if len(feature) > 0:
                    filename += '_' + feature
                filename += '.txt'

                t = extract_keywords_time(filename, 'elapsed_time')
                num = float(t) / 1000.0
                # return the time in second
                print num


def extract_perf_vary_query(datasets=None, queries=None, methods=None, feature=None, input_dir='logs'):
    for query in queries:
        for method in methods:
            print 'query=%s,method=%d' % (query, method)
            for dataset in datasets:
                filename = input_dir + '/' + dataset + \
                    '_a' + str(method) + "_q" + str(query)
                if len(feature) > 0:
                    filename += '_' + feature
                filename += '.txt'

                t = extract_keywords_time(filename, 'elapsed_time')
                num = float(t) / 1000.0
                # return the time in second
                print num


def get_match_count(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
    for dataset in datasets:
        for method in methods:
            print 'dataset=%s,method=%d' % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + "_q" + str(query) + '.txt'
                t = extract_keywords_count(filename, 'total_match_count')
                print t


gpsm_external_phases = ['process_level_time', 'join_phase_time',
                        'build_subgraph_time', 'wait_load_graph_time', 'inspect_join_time']

gpsm_coprocess_phases = ['process_level_time', 'join_phase_time',
                         'build_subgraph_time', 'wait_load_graph_time', 'inspect_join_time']

light_external_phases = ['process_level_time', 'compute_time', 'compute_count_time', 'compute_path_count_time',
                         'materialize_time', 'build_subgraph_time', 'wait_load_graph_time', 'inspect_join_time']
light_coprocessing_inter_partition_phases = [
    'process_level_time', 'compute_time', 'inspect_join_time', 'build_subgraph_time', 'count_time']

reuse_queries = [2, 3, 6, 7, 9, 10, 11, 12]

if __name__ == "__main__":
    # profile external GPU_LIGHT gpu phases
    # profile_phases(['com-youtube.ungraph.txt.bin', 'wiki-Talk.txt.bin', 'com-orkut.ungraph.txt.bin'],
    #               [1, 2, 3, 4, 6, 7], [18], 'gpu_external', light_external_phases, 'logs')

    # profile hybrid coprocessing mode of GPU_LIGHT for inter-partition search
    # profile_phases(['com-youtube.ungraph.txt.bin', 'wiki-Talk.txt.bin'],
    #               [1, 2, 3, 6, 7, 9, 10, 11], [
    #                   18], 'gpu_coprocess_inter_partition_group_inc_load',
    #               light_coprocessing_inter_partition_phases, 'logs')

    # profile coprocessing gpu phases
    #profile_phases(['com-dblp.ungraph.txt.bin','com-youtube.ungraph.txt.bin'], reuse_queries, [15], 'gpu_external', coprocess_phases, 'reuse')

    # extract performance
    extract_perf(['com-lj.ungraph.txt.bin', 'com-orkut.ungraph.txt.bin'],
                 [1, 2, 3, 10, 6, 7], [10], 'gpu_inmemory2', 'logs')
    # extract_perf_vary_query(['com-dblp.ungraph.txt.bin', 'com-youtube.ungraph.txt.bin', 'com-lj.ungraph.txt.bin', 'com-orkut.ungraph.txt.bin', 'uk-2002.txt.bin'],
    #                        [2, 3], [18, 15, 6], 'gpu_inmemory2', 'logs')

    # extract performance for GPU_LIGHT with different variants
    # extract_perf(['com-youtube.ungraph.txt.bin', 'com-lj.ungraph.txt.bin'], [1, 2,
    #                                                                         3, 10, 6, 7, 8, 9, 4], [18], 'gpu_coprocess_inter_partition_ss', 'logs')
    # extract_perf(['com-youtube.ungraph.txt.bin', 'com-lj.ungraph.txt.bin'], [1, 2,
    #                                                                         3, 10, 6, 7, 8, 9, 4], [18], 'gpu_coprocess_inter_partition_group', 'logs')
    # extract_perf(['com-youtube.ungraph.txt.bin', 'wiki-Talk.txt.bin'], [1, 2,
    #                                                                    3, 6, 7, 9, 10, 11], [18], 'gpu_coprocess_inter_partition_group_inc_load', 'logs')
    # extract_perf(['com-friendster.ungraph.txt.bin'], [2,
    #                                                  3, 7], [18], 'gpu_coprocess_inter_partition_group', 'logs')
    # extract_perf(['com-friendster.ungraph.txt.bin'], [2,
    #                                                  3, 7], [18], 'gpu_coprocess_inter_partition_group_inc_load', 'logs')
