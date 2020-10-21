#ifndef __QUERY_OPTIMIZER_TRAVERSAL_PLAN_GENERATOR_H__
#define __QUERY_OPTIMIZER_TRAVERSAL_PLAN_GENERATOR_H__

#include <cassert>
#include "BitSet.h"
#include "Optimizer.h"
#include "QueryCommon.h"

// Generate the traversal plan for one search sequence
class TraversalPlanGenerator {
 public:
  // Try best (to reach a level as large as possible)
  // to choose an order that satisify the connectivity requirement
  // intersect_indices.
  // seq: the currently chosen order.
  // cur_level: current working level
  // intersect_indices: the index-based connectivity for each level,
  // which is the connectivity requirement of the chosen order should satisfy.
  // conn: the connectivity for the original vertex ids
  // status: whether the vertex is used (based on original vertex ids).
  static uintV SetTopoClosestOrder(uintV cur_level, size_t vertex_count,
                                   SearchSequence& seq,
                                   const AllConnType& intersect_indices,
                                   const AllConnType& conn, BitSet& status) {
    if (cur_level == vertex_count) {
      // fully satisify the requirement
      return cur_level;
    }
    auto max_reach_level = cur_level;
    for (uintV u = 0; u < vertex_count; ++u) {
      if (!status.Get(u)) {
        // if u is unused
        // u is connected with some previously visited vertices
        BitSet prev_intersect_indices(vertex_count);
        for (size_t j = 0; j < conn[u].size(); ++j) {
          auto plevel = conn[u][j];
          for (uintV k = 0; k < cur_level; ++k) {
            if (seq[k] == plevel) {
              prev_intersect_indices.Set(k, true);
            }
          }
        }
        // target_indices_status is the connectivity structure
        // we want to achieve
        BitSet target_intersect_indices(vertex_count);
        for (size_t j = 0; j < intersect_indices[cur_level].size(); ++j) {
          size_t plevel_index = intersect_indices[cur_level][j];
          if (plevel_index < cur_level) {
            target_intersect_indices.Set(plevel_index, true);
          }
        }
        if (prev_intersect_indices.Equal(target_intersect_indices)) {
          // if such required status can be achieved
          status.Set(u, true);
          assert(seq.size() > cur_level);
          seq[cur_level] = u;

          auto reach_level =
              SetTopoClosestOrder(cur_level + 1, vertex_count, seq,
                                  intersect_indices, conn, status);
          // choose the path that can reach farthest depth
          // so that the sharing the explored as many levels as possible
          if (reach_level > max_reach_level) {
            // find an order with better sharing score
            max_reach_level = reach_level;
            // Found a topo-equivalent order, terminate
            // At this time, seq stores the match order that can make
            // the topo-equivalent order.
            if (max_reach_level == vertex_count) {
              return max_reach_level;
            }
          }
          // backtrack
          status.Set(u, false);
        }
      }
    }
    return max_reach_level;
  }

  // Given the first cnt vertices decided, choose the order that can
  // maximize the pruning power and store the result in seq.
  // In each level, choose an unused vertex that has maximum connectivity
  // with the chosen vertices.
  static void GeneratePruneOrder(SearchSequence& seq, size_t cnt,
                                 const AllConnType& conn, size_t vertex_count) {
    seq.resize(vertex_count);
    BitSet status(vertex_count);
    for (size_t i = 0; i < cnt; ++i) {
      status.Set(seq[i], true);
    }
    size_t add_count = 0;
    while (1) {
      size_t max_degree = 0;
      size_t c = vertex_count;
      for (uintV i = 0; i < vertex_count; ++i) {
        if (!status.Get(i)) {
          for (size_t j = 0; j < conn[i].size(); ++j) {
            auto k = conn[i][j];
            if (status.Get(k)) {
              size_t d = conn[i].size();
              if (max_degree < d) {
                max_degree = d;
                c = i;
                break;
              }
            }
          }
        }
      }
      if (max_degree == 0) break;
      seq[cnt + add_count] = c;
      ++add_count;
      status.Set(c, true);
    }
    assert(cnt + add_count == vertex_count);
  }

  // validate whether two dfs orders are equivalent
  static bool TopoEquivalent(SearchSequence& seq1, SearchSequence& seq2,
                             const AllConnType& conn, size_t vertex_count) {
    for (uintV l = 0; l < vertex_count; ++l) {
      if (conn[seq1[l]].size() != conn[seq2[l]].size()) {
        return false;
      }
      for (size_t j = 0; j < conn[seq1[l]].size(); ++j) {
        auto plevel = conn[seq1[l]][j];
        // in seq1, there is such predicate (seq1[l] -> plevel)
        // in view of index, it is l -> prev_index1
        // topo equivalence requires l -> prev_index2(=prev_index1) for seq2 as
        // well
        auto prev_index1 = vertex_count;
        for (uintV k = 0; k < vertex_count; ++k) {
          if (plevel == seq1[k]) {
            prev_index1 = k;
            break;
          }
        }
        bool exists = false;
        for (size_t k = 0; k < conn[seq2[l]].size(); ++k) {
          if (conn[seq2[l]][k] == seq2[prev_index1]) {
            exists = true;
            break;
          }
        }
        if (!exists) return false;
      }
    }
    return true;
  }
};

class TraversalPlanGenerateHelper {
 public:
  // Given the search order seq, transform the original connectivity to be based
  // on the index of seq
  static void GetIndexBasedConnectivity(AllConnType& intersect_indices,
                                        const AllConnType& con,
                                        const SearchSequence& seq,
                                        size_t vertex_count) {
    std::vector<uintV> nto_map(vertex_count);
    for (uintV u = 0; u < vertex_count; ++u) {
      auto nu = seq[u];
      nto_map[nu] = u;
    }

    intersect_indices.resize(vertex_count);
    for (uintV u = 0; u < vertex_count; ++u) {
      auto nu = seq[u];
      intersect_indices[u].assign(con[nu].begin(), con[nu].end());
      for (size_t j = 0; j < intersect_indices[u].size(); ++j) {
        auto nv = intersect_indices[u][j];
        intersect_indices[u][j] = nto_map[nv];
      }
    }
  }

  // Similar to GetIndexBasedConnectivity. But transform the connectivity for
  // one edge list and store the result in status
  static void GetIndexBasedConnectivity(BitSet& status,
                                        const std::vector<uintV>& con,
                                        const SearchSequence& seq,
                                        size_t vertex_count) {
    std::vector<uintV> nto_map(vertex_count);
    for (uintV u = 0; u < vertex_count; ++u) {
      auto nu = seq[u];
      nto_map[nu] = u;
    }

    for (size_t j = 0; j < con.size(); ++j) {
      auto nv = con[j];
      status.Set(nto_map[nv], true);
    }
  }

  // from multiple groups of dfs ids, i.e., group_dfs_ids,
  // extract the group id for each dfs and store the result in dfs_group_id
  static void GetDfsGroupId(std::vector<uintV>& dfs_group_id,
                            MultiDfsIdGroup& group_dfs_ids) {
    size_t group_num = group_dfs_ids.size();
    for (size_t group_id = 0; group_id < group_num; ++group_id) {
      for (size_t gidx = 0; gidx < group_dfs_ids[group_id].size(); ++gidx) {
        size_t dfs_id = group_dfs_ids[group_id][gidx];
        dfs_group_id[dfs_id] = group_id;
      }
    }
  }

  // for multiple groups of dfs ids, evaluate whether dfs_ids1 and dfs_ids2
  // are in the same group somewhere
  static bool SameGroup(MultiDfsIdGroup& groups, size_t dfs_id1,
                        size_t dfs_id2) {
    size_t group_num = groups.size();
    for (size_t group_id = 0; group_id < group_num; ++group_id) {
      DfsIdGroup& group = groups[group_id];
      bool flag1 = false;
      bool flag2 = false;
      for (auto& dfs_id : group) {
        if (dfs_id == dfs_id1) flag1 = true;
        if (dfs_id == dfs_id2) flag2 = true;
      }
      if (flag1 && flag2) {
        return true;
      }
    }
    return false;
  }
};

#endif
