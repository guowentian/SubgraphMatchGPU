#ifndef __GPU_COMPONENT_CHECK_CONSTRAINTS_REUSE_COMMON_CUH__
#define __GPU_COMPONENT_CHECK_CONSTRAINTS_REUSE_COMMON_CUH__

#include "DevReuseTraversalPlan.cuh"
// This file provides some functionality to support reuse when checking
// constraints. E.g., some functions used by local threads

// Given the partial instance M, a thread find the cached_offset,
// based on which, we can retrieve the cached intersection result.
// Note that cached_offset is the corresponding position of d_next_offsets
// (minus the batch offset).
// The partial instances are represented by (d_instances, d_parents_indices)
// like a prefix tree.
DEVICE bool FastLocateCache(uintV* M, size_t cur_index, size_t cur_level,
                            size_t aligned_level, size_t target_level,
                            uintV** d_instances, size_t** d_parents_indices,
                            size_t** d_next_offsets,
                            size_t* level_batch_offsets_start,
                            size_t* level_batch_offsets_end,
                            size_t& cached_offset) {
  assert(aligned_level <= target_level);
  assert(target_level > 0);

  // 1. move backward to aligned_level
  while (cur_level > aligned_level) {
    cur_index = d_parents_indices[cur_level][cur_index];
    --cur_level;
  }

  // If we have reached target_level, can directly return,
  // Otherwise, we need a process to move forward .
  if (cur_level == target_level) {
    // Since target_level>0, d_parents_indices[target_level] must exist
    cur_index = d_parents_indices[cur_level][cur_index];
#if defined(DEBUG)
    // Ensure the position to access d_next_offsets are in the range of batching
    assert(cur_index >= level_batch_offsets_start[target_level - 1]);
    assert(cur_index <= level_batch_offsets_end[target_level - 1]);
#endif
    cached_offset = cur_index - level_batch_offsets_start[target_level - 1];

    return true;
  }

  // 2. We may need to move one more previous level.
  // From there, we start to move forward to binary search the corresponding
  // position.
  size_t search_array_start_index;
  size_t search_array_size;
  if (cur_level == 0) {
    // The whole array of instances at level 0 is searched
    search_array_start_index = 0;
    search_array_size =
        level_batch_offsets_end[0] - level_batch_offsets_start[0];
  } else {
    cur_index = d_parents_indices[cur_level][cur_index];
    // Now cur_index is the position of the instances at cur_level-1.
    // But to access d_next_offsets, we need the offset that are in batch of
    // cur_level-1.
    assert(cur_index >= level_batch_offsets_start[cur_level - 1]);
    size_t batch_index = cur_index - level_batch_offsets_start[cur_level - 1];
    search_array_start_index = d_next_offsets[cur_level - 1][batch_index];
    search_array_size = d_next_offsets[cur_level - 1][batch_index + 1] -
                        d_next_offsets[cur_level - 1][batch_index];
  }

  // 3. For array [d_instances[cur_level] + search_array_start_index,
  // d_instances[cur_level] + search_array_start_index + search_array_size),
  // we will search whether M[cur_level] exists in such an array.
  // Repeating this, we can move forward to find the cached_offset

  size_t prev_find_index = cur_index;
  while (cur_level < target_level) {
    uintV* search_array = d_instances[cur_level] + search_array_start_index;
    uintV search_element = M[cur_level];
    size_t find_pos = GpuUtils::BinSearch::BinSearch(
        search_array, search_array_size, search_element);
    cur_index = search_array_start_index + find_pos;

    bool found = false;
    if (find_pos < search_array_size &&
        search_array[find_pos] == search_element) {
      // if this position is in the batching range and the child of this path is
      // available.
      if (level_batch_offsets_start[cur_level] <= cur_index &&
          cur_index < level_batch_offsets_end[cur_level]) {
        found = true;
      }
    }

    if (found) {
      assert(cur_index >= level_batch_offsets_start[cur_level]);
      size_t batch_index = cur_index - level_batch_offsets_start[cur_level];
      search_array_start_index = d_next_offsets[cur_level][batch_index];
      search_array_size = d_next_offsets[cur_level][batch_index + 1] -
                          d_next_offsets[cur_level][batch_index];
      prev_find_index = cur_index;
    } else {
      // Cannot find the required cached result
      return false;
    }

    ++cur_level;
  }
#if defined(DEBUG)
  // Ensure the position to access d_next_offsets are in the range of batching
  assert(prev_find_index >= level_batch_offsets_start[target_level - 1]);
  assert(prev_find_index <= level_batch_offsets_end[target_level - 1]);
#endif
  cached_offset = prev_find_index - level_batch_offsets_start[target_level - 1];
  return true;
}

// FindCachedOffset is used by each thread to find the positions of all cached
// results. Return true if all cached results can be found. The positions of
// cached results are written to reuse_indices.
DEVICE bool ThreadFindCachedOffset(
    DevVertexReuseIntersectPlan& vertex_plan, uintV* M, size_t last_level_index,
    size_t last_level, uintV** d_instances, size_t** d_parents_indices,
    size_t** d_next_offsets, size_t* level_batch_offsets_start,
    size_t* level_batch_offsets_end, size_t path_id, size_t** reuse_indices) {
  size_t reuse_conn_meta_count = vertex_plan.GetReuseConnectivityMetaCount();
  // If there is no reuse according to the plan
  if (reuse_conn_meta_count == 0) {
    return false;
  }
  bool valid = true;
  for (size_t i = 0; i < reuse_conn_meta_count; ++i) {
    auto& reuse_conn_meta = vertex_plan.GetReuseConnectivityMeta(i);
    uintV aligned_level = reuse_conn_meta.GetAlignedVertex();
    uintV target_level = reuse_conn_meta.GetSourceVertex();

    // Build the partial instance used to retrieve the position of cached
    // result
    uintV new_M[kMaxQueryVerticesNum];
    auto& conn = reuse_conn_meta.GetConnectivity();
    auto& source_conn = reuse_conn_meta.GetSourceConnectivity();
    for (size_t j = 0; j < conn.GetCount(); ++j) {
      new_M[source_conn.GetConnectivity(j)] = M[conn.GetConnectivity(j)];
    }

    size_t cached_offset = 0;
    bool found = FastLocateCache(
        new_M, last_level_index, last_level, aligned_level, target_level,
        d_instances, d_parents_indices, d_next_offsets,
        level_batch_offsets_start, level_batch_offsets_end, cached_offset);

    if (found) {
      reuse_indices[i][path_id] = cached_offset;
    } else {
      valid = false;
      break;
    }
  }
  return valid;
}

// Check whether 'candidate' on 'path_id' satisfy the connectivity requirement.
// First is the membership test on the remaining cached result, and then
// test on remaining separate connectivity.
DEVICE bool ThreadCheckConnectivityReuse(
    DevVertexReuseIntersectPlan& vertex_plan, uintV* M, size_t path_id,
    uintV candidate, size_t** d_seq_reuse_indices_data,
    size_t** d_seq_next_offsets_data, uintV** d_seq_cached_results_data,
    uintE* row_ptrs_data, uintV* cols_data) {
  bool valid = true;
  size_t reuse_conn_meta_count = vertex_plan.GetReuseConnectivityMetaCount();
  for (size_t i = 1; i < reuse_conn_meta_count; ++i) {
    auto& reuse_conn_meta = vertex_plan.GetReuseConnectivityMeta(i);
    uintV target_level = reuse_conn_meta.GetSourceVertex();
    size_t cached_offset = d_seq_reuse_indices_data[i][path_id];
    uintV* array = d_seq_cached_results_data[target_level] +
                   d_seq_next_offsets_data[target_level - 1][cached_offset];
    size_t array_size =
        d_seq_next_offsets_data[target_level - 1][cached_offset + 1] -
        d_seq_next_offsets_data[target_level - 1][cached_offset];
    size_t find_pos =
        GpuUtils::BinSearch::BinSearch(array, array_size, candidate);
    if (!(find_pos < array_size && array[find_pos] == candidate)) {
      valid = false;
      break;
    }
  }

  if (valid) {
    DevConnType& separate_conn = vertex_plan.GetSeparateConnectivity();
    for (size_t i = 0; i < separate_conn.GetCount(); ++i) {
      uintV u = separate_conn.GetConnectivity(i);
      uintV v = M[u];
      uintV* search_array = cols_data + row_ptrs_data[v];
      size_t search_array_size = row_ptrs_data[v + 1] - row_ptrs_data[v];
      size_t find_pos = GpuUtils::BinSearch::BinSearch(
          search_array, search_array_size, candidate);
      if (!(find_pos < search_array_size &&
            search_array[find_pos] == candidate)) {
        valid = false;
        break;
      }
    }
  }
  return valid;
}

DEVICE void ThreadChoosePivotIndexReuse(
    DevVertexReuseIntersectPlan& reuse_intersect_plan, uintV* M, size_t path_id,
    size_t** d_seq_reuse_indices_data, size_t** d_seq_cache_next_offsets_data,
    uintE* row_ptrs_data, bool& pivot_from_cache, uintV& pivot_index) {
  pivot_from_cache = false;
  pivot_index = 0;
  size_t candidates_count = kMaxsize_t;
  size_t reuse_conn_meta_count =
      reuse_intersect_plan.GetReuseConnectivityMetaCount();
  for (size_t j = 0; j < reuse_conn_meta_count; ++j) {
    auto& reuse_conn_meta = reuse_intersect_plan.GetReuseConnectivityMeta(j);
    size_t cached_offset = d_seq_reuse_indices_data[j][path_id];
    uintV target_level = reuse_conn_meta.GetSourceVertex();
    size_t cur_candidates_count =
        d_seq_cache_next_offsets_data[target_level - 1][cached_offset + 1] -
        d_seq_cache_next_offsets_data[target_level - 1][cached_offset];
    if (cur_candidates_count < candidates_count) {
      pivot_from_cache = true;
      pivot_index = j;
      candidates_count = cur_candidates_count;
    }
  }

  auto& separate_conn = reuse_intersect_plan.GetSeparateConnectivity();
  for (size_t j = 0; j < separate_conn.GetCount(); ++j) {
    uintV u = separate_conn.Get(j);
    uintV v = M[u];
    size_t adj_count = row_ptrs_data[v + 1] - row_ptrs_data[v];
    if (adj_count < candidates_count) {
      pivot_from_cache = false;
      pivot_index = j;
      candidates_count = adj_count;
    }
  }
#if defined(DEBUG)
  assert(candidates_count < kMaxsize_t);
#endif
}

DEVICE bool ThreadCheckConnectivityReuseOpt(
    DevVertexReuseIntersectPlan& vertex_plan, uintV* M, size_t path_id,
    uintV candidate, size_t** d_seq_reuse_indices_data,
    size_t** d_seq_next_offsets_data, uintV** d_seq_cached_results_data,
    uintE* row_ptrs_data, uintV* cols_data, bool pivot_from_cache,
    uintV pivot_index) {
  bool valid = true;
  size_t reuse_conn_meta_count = vertex_plan.GetReuseConnectivityMetaCount();
  for (size_t i = 0; i < reuse_conn_meta_count; ++i) {
    if (!(pivot_from_cache && pivot_index == i)) {
      auto& reuse_conn_meta = vertex_plan.GetReuseConnectivityMeta(i);
      uintV target_level = reuse_conn_meta.GetSourceVertex();
      size_t cached_offset = d_seq_reuse_indices_data[i][path_id];
      uintV* array = d_seq_cached_results_data[target_level] +
                     d_seq_next_offsets_data[target_level - 1][cached_offset];
      size_t array_size =
          d_seq_next_offsets_data[target_level - 1][cached_offset + 1] -
          d_seq_next_offsets_data[target_level - 1][cached_offset];
      size_t find_pos =
          GpuUtils::BinSearch::BinSearch(array, array_size, candidate);
      if (!(find_pos < array_size && array[find_pos] == candidate)) {
        valid = false;
        break;
      }
    }
  }

  if (valid) {
    DevConnType& separate_conn = vertex_plan.GetSeparateConnectivity();
    for (size_t i = 0; i < separate_conn.GetCount(); ++i) {
      if (!(!pivot_from_cache && i == pivot_index)) {
        uintV u = separate_conn.GetConnectivity(i);
        uintV v = M[u];
        uintV* search_array = cols_data + row_ptrs_data[v];
        size_t search_array_size = row_ptrs_data[v + 1] - row_ptrs_data[v];
        size_t find_pos = GpuUtils::BinSearch::BinSearch(
            search_array, search_array_size, candidate);
        if (!(find_pos < search_array_size &&
              search_array[find_pos] == candidate)) {
          valid = false;
          break;
        }
      }
    }
  }
  return valid;
}

#endif
