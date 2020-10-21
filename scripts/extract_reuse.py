from extract_base import extract_keywords_time
from extract_base import extract_keywords_multi_time
from extract_base import extract_keywords_count


def exp_effect_reuse(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
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


def get_match_count(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
    for dataset in datasets:
        for method in methods:
            print 'dataset=%s,method=%d' % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + "_q" + str(query) + '.txt'
                t = extract_keywords_count(filename, 'total_match_count')
                print t


def profile_cpu_check_connectivity_ratio(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
    for dataset in datasets:
        for method in methods:
            print 'dataset=%s,method=%d, intersect_time/(intersect_time+condition_time)' % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                t = extract_keywords_count(
                    filename, 'check_connectivity/check_constraints')
                print t
            print 'dataset=%s, intersect_time/elapsed_time' % dataset
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                connectivity_time = extract_keywords_time(
                    filename, 'intersect_time')
                elapsed_time = extract_keywords_time(filename, 'elapsed_time')
                ratio = float(connectivity_time) / float(elapsed_time)
                # print connectivity_time + ',' + elapsed_time + ',' + str(ratio)
                print ratio


def profile_cpu_reuse(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
    for dataset in datasets:
        for method in methods:
            print "dataset=%s,method=%d, reuse ratio" % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                reuse_ratio = extract_keywords_count(filename, 'reuse ratio')
                print reuse_ratio
            print "dataset=%s, reuse intersect result size versus candidates size" % dataset
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                ratio = extract_keywords_count(
                    filename, 'reuse intersect result size versus candidates size')
                print ratio


def profile_gpu_memory_operation_overheads(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
    for dataset in datasets:
        for method in methods:
            print 'dataset=%s,method=%d, memory_operations_time' % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                t = extract_keywords_time(filename, 'memory_operations_time')
                print t
            print 'dataset=%s, elapsed_time' % dataset
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                t = extract_keywords_time(filename, 'elapsed_time')
                print t


def profile_gpu_check_connectivity_ratio(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
    for dataset in datasets:
        for method in methods:
            print 'dataset=%s,method=%d, intersect_time/(predicate_time+condition_time)' % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                intersect_time = extract_keywords_time(
                    filename, 'intersect_time')
                predicate_time = extract_keywords_time(
                    filename, 'predicate_time')
                condition_time = extract_keywords_time(
                    filename, 'condition_time')
                ratio = float(intersect_time) / \
                    (float(predicate_time) + float(condition_time))
                print ratio

            print 'dataset=%s,predicate_time/(predicate_time+condition_time)' % dataset
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                predicate_time = extract_keywords_time(
                    filename, 'predicate_time')
                condition_time = extract_keywords_time(
                    filename, 'condition_time')
                ratio = float(predicate_time) / \
                    (float(predicate_time) + float(condition_time))
                print ratio

            print 'dataset=%s,intersect_time/elapsed_time' % dataset
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                intersect_time = extract_keywords_time(
                    filename, 'intersect_time')
                elapsed_time = extract_keywords_time(filename, 'elapsed_time')
                ratio = float(intersect_time) / float(elapsed_time)
                print ratio


def profile_gpu_reuse(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
    time_keywords = ['process_level_time', 'check_order_non_equality_time', 'check_connectivity_time', 'organize_batch_time', 'reuse_overhead_time',
                     'predicate_time', 'intersect_time', 'compact_time', 'iterate_gather_time', 'load_balance_search_time', 'memory_operations_time']

    for dataset in datasets:
        for method in methods:
            print "dataset=%s,method=%d, in ms" % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                time_collection = []
                for keyword in time_keywords:
                    t = extract_keywords_time(filename, keyword)
                    time_collection.append(t)
                ss = ''
                for i in range(len(time_collection)):
                    if i > 0:
                        ss += ','
                    ss += time_collection[i]
                print ss


def profile_gpu_set_intersect_time_ratio(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
    for dataset in datasets:
        for method in methods:
            print "dataset=%s,method=%d, in ms" % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                set_intersect_time = extract_keywords_time(
                    filename, "check_constraints_time")
                total_time = extract_keywords_time(filename, "elapsed_time")
                ratio = float(set_intersect_time) / float(total_time)
                # print "%s,%s" % (set_intersect_time, total_time)
                # print set_intersect_time
                print total_time
                # print ratio


def profile_gpu_reuse_ratio(datasets=None, queries=None, methods=None, feature=None, input_dir='reuse'):
    for dataset in datasets:
        for method in methods:
            print "dataset=%s,method=%d" % (dataset, method)
            for query in queries:
                filename = input_dir + '/' + dataset + '_a' + \
                    str(method) + '_q' + str(query) + '_' + feature + '.txt'
                reuse_count = extract_keywords_count(filename, 'reuse_count')
                intersect_count = extract_keywords_count(
                    filename, 'intersect_count')
                ratio = float(reuse_count) / float(intersect_count)
                # print str(reuse_count) + "," + str(intersect_count) + "," + str(ratio)
                print ratio


all_datasets = ['com-youtube.ungraph.txt.bin', 'wiki-Talk.txt.bin', 'wiki-topcats.txt.bin',
                'soc-twitter-higgs.txt.bin', 'soc-dogster.txt.bin', 'com-orkut.ungraph.txt.bin']
all_queries = [2, 3, 6, 7, 9, 10, 11, 12]

gpu_methods = [15, 16, 17, 6]
gpu_perf_feature = 'gpu_cnmem2_profile'
gpu_profile_feature = 'gpu_cnmem2_reuse_profile'

cpu_methods = [8, 9]
cpu_perf_feature = 'cpu'


if __name__ == '__main__':
    exp_effect_reuse(all_datasets, [0, 1], [4, 5], gpu_perf_feature)
    #exp_effect_reuse(['soc-dogster.txt.bin'], all_queries, [8,9], cpu_perf_feature)
    # get_match_count()

    # profile_gpu_memory_operation_overheads()
    # profile_gpu_check_connectivity_ratio()
    # profile_cpu_check_connectivity_ratio()
    # profile_cpu_reuse()
    # profile_gpu_reuse()
    # profile_gpu_set_intersect_time_ratio()
    # profile_gpu_reuse_ratio()
