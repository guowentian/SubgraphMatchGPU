#ifndef __HYBRID_GPU_COMPONENT_CHECK_CONSTRAINTS_COMMON_CUH__
#define __HYBRID_GPU_COMPONENT_CHECK_CONSTRAINTS_COMMON_CUH__

#include "DevTraversalPlan.cuh"
#include "GPUBinSearch.cuh"
#include "GPUFilter.cuh"

// For a set of connectivity as in 'conn', find the level that has the
// smallest adjacent list.
DEVICE uintV ThreadChoosePivotLevel(DevConnType& conn, uintV* M,
                                    uintE* row_ptrs_data) {
  uintV pivot_level = 0;
  size_t pivot_adj_count = 0;
  for (size_t i = 0; i < conn.GetCount(); ++i) {
    uintV u = conn.GetConnectivity(i);
    uintV v = M[u];
    size_t adj_count = row_ptrs_data[v + 1] - row_ptrs_data[v];
    if (i == 0) {
      pivot_level = u;
      pivot_adj_count = adj_count;
    } else {
      if (pivot_adj_count > adj_count) {
        pivot_adj_count = adj_count;
        pivot_level = u;
      }
    }
  }
  return pivot_level;
}
// Check whetehr 'candidate' on 'path_id' satisfy the connecitivity requirement.
// Normal execution path.
DEVICE bool ThreadCheckConnectivity(DevConnType& conn, uintV* M,
                                    uintV candidate, char pivot_level,
                                    uintE* row_ptrs_data, uintV* cols_data) {
  DevConnType check_conn;
  check_conn.count_ = 0;
  for (size_t i = 0; i < conn.GetCount(); ++i) {
    uintV u = conn.GetConnectivity(i);
    if (u != pivot_level) {
      check_conn.array_[check_conn.count_++] = u;
    }
  }

  for (size_t i = 0; i < check_conn.GetCount(); ++i) {
    uintV u = check_conn.GetConnectivity(i);
    uintV v = M[u];
    uintV* search_array = cols_data + row_ptrs_data[v];
    size_t search_array_size = row_ptrs_data[v + 1] - row_ptrs_data[v];
    size_t find_pos = GpuUtils::BinSearch::BinSearch(
        search_array, search_array_size, candidate);
    if (!(find_pos < search_array_size &&
          search_array[find_pos] == candidate)) {
      return false;
    }
  }
  return true;
}

DEVICE bool ThreadCheckCondition(DevCondArrayType& cond, uintV* M,
                                 uintV candidate) {
  for (size_t i = 0; i < cond.GetCount(); ++i) {
    DevCondType c = cond.GetCondition(i);
    if (!GpuUtils::Filter::CheckVertexCondition(candidate, M[c.GetOperand()],
                                                c.GetOperator())) {
      return false;
    }
  }
  return true;
}
// Return true if no duplicates, e1 < e2
// Return false if there is duplicates, e1 >= e2
DEVICE bool ThreadCheckDuplicate(uintV e1u, uintV e1v, uintV e2u, uintV e2v) {
  IndexTypeTuple2 e1;
  e1.x = Min(e1u, e1v);
  e1.y = Max(e1u, e1v);
  IndexTypeTuple2 e2;
  e2.x = Min(e2u, e2v);
  e2.y = Max(e2u, e2v);
  // ensure e1 < e2
  if (e1.x < e2.x || (e1.x == e2.x && e1.y < e2.y)) {
    return true;
  }
  return false;
}

DEVICE bool ThreadCheckDuplicate(DevConnType& conn, uintV* M, uintV candidate,
                                 uintP* d_partition_ids) {
  for (size_t i = 0; i < conn.GetCount(); ++i) {
    uintV u = conn.Get(i);
    uintV v = M[u];
    if (d_partition_ids[candidate] != d_partition_ids[v]) {
#if defined(DEBUG)
      assert(d_partition_ids[M[0]] != d_partition_ids[M[1]]);
#endif
      bool f = ThreadCheckDuplicate(M[0], M[1], v, candidate);
      if (!f) {
        return false;
      }
    }
  }
  return true;
}

DEVICE bool ThreadCheckDuplicate(DevConnType& conn, uintV* M, uintV candidate,
                                 uintP* d_partition_ids, uintV prime_edge_v0,
                                 uintV prime_edge_v1) {
  for (size_t i = 0; i < conn.GetCount(); ++i) {
    uintV u = conn.Get(i);
    uintV v = M[u];
    if (d_partition_ids[candidate] != d_partition_ids[v]) {
#if defined(DEBUG)
      assert(d_partition_ids[M[prime_edge_v0]] !=
             d_partition_ids[M[prime_edge_v1]]);
#endif
      bool f = ThreadCheckDuplicate(M[prime_edge_v0], M[prime_edge_v1], v,
                                    candidate);
      if (!f) {
        return false;
      }
    }
  }
  return true;
}

// Used when directly counting the #instances for two
// unmaterialized pattern vertices u1, u2.
// Given u1 that has been computed, for each candidate of u2,
// i.e., search_element,
// count the corresponding #candidates of u1.
// u2(search_element) (cond_operator) u1(search_array)
DEVICE size_t ThreadCountWithComputedVertex(uintV* search_array,
                                            size_t search_count,
                                            uintV search_element,
                                            CondOperator cond_operator) {
  if (search_count == 0) {
    return 0;
  }
  size_t bin_pos = GpuUtils::BinSearch::BinSearch(search_array, search_count,
                                                  search_element);
  bool equal =
      (bin_pos < search_count && search_array[bin_pos] == search_element);

  size_t ret = 0;
  if (cond_operator == LESS_THAN) {
    bin_pos += equal ? 1 : 0;
    ret = search_count - bin_pos;
  } else if (cond_operator == LARGER_THAN) {
    ret = bin_pos;
  } else {
    // NON_EQUAL
    ret = search_count;
    ret -= equal ? 1 : 0;
  }
  return ret;
}

// Used for OrganizeBatch.
// To estimate the potential memory cost for each path, calculate the
// number of expanded children (MIN);
// calculate the size of adjacent lists used to expand a new level (ADD).
template <OperatorType op>
DEVICE size_t ThreadEstimatePathCount(DevConnType& conn, uintV* M,
                                      uintE* row_ptrs) {
  size_t ret;
  if (op == ADD) {
    ret = 0;
  } else if (op == MAX) {
    ret = 0;
  } else if (op == MIN) {
    ret = kMaxsize_t;
  } else {
    assert(false);
  }

  for (size_t i = 0; i < conn.GetCount(); ++i) {
    uintV u = conn.Get(i);
    uintV v = M[u];
    size_t count = row_ptrs[v + 1] - row_ptrs[v];
    if (op == ADD) {
      ret += count;
    } else if (op == MAX) {
      ret = Max(ret, count);
    } else if (op == MIN) {
      ret = Min(ret, count);
    }
  }

  return ret;
}

#endif
