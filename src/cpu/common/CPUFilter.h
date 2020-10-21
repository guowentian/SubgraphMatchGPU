#ifndef __CPU_FILTER_H__
#define __CPU_FILTER_H__

#include "Meta.h"
#include "Query.h"

static bool CheckCondition(uintV *path, uintV target,
                           std::vector<CondType> &cond) {
  for (size_t j = 0; j < cond.size(); ++j) {
    switch (cond[j].first) {
      case LESS_THAN:
        if (!(target < path[cond[j].second])) return false;
        break;
      case LARGER_THAN:
        if (!(target > path[cond[j].second])) return false;
        break;
      case NON_EQUAL:
        if (target == path[cond[j].second]) return false;
        break;
      default:
        break;
    }
  }
  return true;
}
template <typename IndexType>
static bool CheckEquality(IndexType *path, size_t len, IndexType target) {
  for (size_t i = 0; i < len; ++i) {
    if (path[i] == target) {
      return true;
    }
  }
  return false;
}
template <typename IndexType>
static bool CheckEquality(IndexType *path, size_t len, IndexType target,
                          std::vector<IndexType> &order) {
  for (size_t i = 0; i < len; ++i) {
    if (path[order[i]] == target) {
      return true;
    }
  }
  return false;
}

// return true if e1 <= e2
template <typename IndexType>
static bool ComparePatternEdgeOrder(std::pair<IndexType, IndexType> &e1,
                                    std::pair<IndexType, IndexType> &e2) {
  if (e1.first < e2.first || (e1.first == e2.first && e1.second <= e2.second))
    return true;
  return false;
}
// Return true if there is no duplicate
static bool CheckDuplicate(uintV *path, uintV target,
                           std::vector<uintV> &intersect_levels,
                           uintP *partition_ids, uintV prime_edge_0 = 0,
                           uintV prime_edge_1 = 1) {
  auto e1 = std::make_pair(std::min(path[prime_edge_0], path[prime_edge_1]),
                           std::max(path[prime_edge_0], path[prime_edge_1]));
  auto cur_p = partition_ids[target];
  for (size_t j = 0; j < intersect_levels.size(); ++j) {
    auto idx = intersect_levels[j];
    auto prev_p = partition_ids[path[idx]];
    if (cur_p != prev_p) {
      // cross-partition edges
      auto e2 = std::make_pair(std::min(path[idx], target),
                               std::max(path[idx], target));
      if (!ComparePatternEdgeOrder(e1, e2)) {
        return false;
      }
    }
  }
  return true;
}

#endif
