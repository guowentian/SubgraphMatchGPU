#ifndef __CPU_PARALLEL_INTERSECTION_H__
#define __CPU_PARALLEL_INTERSECTION_H__

#include <vector>
#include "CPUIntersection.h"
#include "ParallelUtils.h"

template <typename IndexType, typename DataType>
static void ParallelIntersect(std::vector<DataType*>& edges1_start_ptrs,
                              std::vector<DataType*>& edges1_end_ptrs,
                              std::vector<DataType*>& edges2_start_ptrs,
                              std::vector<DataType*>& edges2_end_ptrs,
                              size_t path_num,
                              std::vector<IndexType>& output_row_ptrs,
                              std::vector<DataType>& output_inst) {
  // obtain size of candidates for each path
  std::vector<IndexType> candidates_count(path_num);
  parallel_for(IndexType path_id = 0; path_id < path_num; ++path_id) {
    DataType* edges1_start_ptr = edges1_start_ptrs[path_id];
    DataType* edges1_end_ptr = edges1_end_ptrs[path_id];
    DataType* edges2_start_ptr = edges2_start_ptrs[path_id];
    DataType* edges2_end_ptr = edges2_end_ptrs[path_id];
    candidates_count[path_id] = std::min(edges1_end_ptr - edges1_start_ptr,
                                         edges2_end_ptr - edges2_start_ptr);
  }

  // scan
  std::vector<IndexType> candidates_offset(path_num + 1);
  IndexType total_candidates_count = ParallelUtils::ParallelPlusScan(
      candidates_count.data(), candidates_offset.data(), path_num);
  candidates_offset[path_num] = total_candidates_count;

  /*#if defined(DEBUG)
    for (IndexType path_id = 0; path_id < path_num; ++path_id) {
      long long v = candidates_offset[path_id];
      assert(v + candidates_count[path_id] == candidates_offset[path_id + 1]);
    }
  #endif
  */

  // allocate output_inst
  std::vector<IndexType>& candidates_output_count = candidates_count;
  std::vector<IndexType> candidates_output_inst(total_candidates_count);

  // for each path, intersect, and indicate the satisfied count
  parallel_for(IndexType path_id = 0; path_id < path_num; ++path_id) {
    DataType* edges1_start_ptr = edges1_start_ptrs[path_id];
    DataType* edges1_end_ptr = edges1_end_ptrs[path_id];
    DataType* edges2_start_ptr = edges2_start_ptrs[path_id];
    DataType* edges2_end_ptr = edges2_end_ptrs[path_id];
    DataType* output_ptr =
        candidates_output_inst.data() + candidates_offset[path_id];

    size_t res_size =
        SortedIntersection(edges1_start_ptr, edges1_end_ptr, edges2_start_ptr,
                           edges2_end_ptr, output_ptr);
    candidates_output_count[path_id] = res_size;
  }

  // scan the satisfied count, resize output_inst
  output_row_ptrs.resize(path_num + 1);
  IndexType total_valid_count = ParallelUtils::ParallelPlusScan(
      candidates_output_count.data(), output_row_ptrs.data(), path_num);
  output_row_ptrs[path_num] = total_valid_count;
  output_inst.resize(total_valid_count);

  // pack the valid candidates into output_inst
  parallel_for(IndexType path_id = 0; path_id < path_num; ++path_id) {
    DataType* candidates = &candidates_output_inst[candidates_offset[path_id]];
    IndexType count = output_row_ptrs[path_id + 1] - output_row_ptrs[path_id];
    for (IndexType j = 0; j < count; ++j) {
      output_inst[output_row_ptrs[path_id] + j] = *(candidates + j);
    }
  }
}

#endif
