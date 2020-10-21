#ifndef __QUERY_COMMON_MULTI_TRAVERSAL_PLAN_GENERATOR_H__
#define __QUERY_COMMON_MULTI_TRAVERSAL_PLAN_GENERATOR_H__

#include "TraversalPlanGenerator.h"

#include <algorithm>

// generate the traversal plan for multiple search sequences
// (inter-partition).
class MultiTraversalPlanGenerator {
 public:
  // ========= make topo-equivalent search sequences in the same group
  // write results to group_dfs_ids and dfs_orders
  // dfs_orders: the search sequences for each dfs
  // group_dfs_ids: the dfs ids for each equivalent group
  static void GenerateTopoEquivalentOrder(
      std::vector<SearchSequence>& dfs_orders, MultiDfsIdGroup& group_dfs_ids,
      const AllConnType& conn, size_t vertex_count) {
    // initialize the first two vertices
    for (uintV l = 0; l < vertex_count; ++l) {
      for (size_t pred_id = 0; pred_id < conn[l].size(); ++pred_id) {
        auto plevel = conn[l][pred_id];
        // pattern edge l->plevel
        if (l < plevel) {
          SearchSequence order(vertex_count);
          order[0] = l;
          order[1] = plevel;
          dfs_orders.push_back(order);
        }
      }
    }

    std::vector<size_t> dfs_group_id(dfs_orders.size());
    BitSet dfs_vis(dfs_orders.size());
    size_t cur_group_id = 0;
    for (size_t didx = 0; didx < dfs_orders.size(); ++didx) {
      if (!dfs_vis.Get(didx)) {
        dfs_vis.Set(didx, true);
        // if this dfs order is not finished
        // given the first edge, generate an order with pruning
        TraversalPlanGenerator::GeneratePruneOrder(dfs_orders[didx], 2, conn,
                                                   vertex_count);
        dfs_group_id[didx] = cur_group_id++;

        // obtain the index version of dfs_orders[didx], which is used to
        // generate other topo equivalent dfs orders
        AllConnType dfs_intersect_indices;
        TraversalPlanGenerateHelper::GetIndexBasedConnectivity(
            dfs_intersect_indices, conn, dfs_orders[didx], vertex_count);

        // try to generate other orders that are topo equivalent to
        // dfs_order_group[didx]
        for (size_t i = didx + 1; i < dfs_orders.size(); ++i) {
          if (!dfs_vis.Get(i)) {
            // for other unprocessed dfs order sequences, try best to make it to
            // be topo-equivalent to dfs_orders[didx]
            BitSet status(vertex_count);
            status.Set(dfs_orders[i][0], true);
            status.Set(dfs_orders[i][1], true);
            if (TraversalPlanGenerator::SetTopoClosestOrder(
                    2, vertex_count, dfs_orders[i], dfs_intersect_indices, conn,
                    status) == vertex_count) {
              // found an order that can be topo- equivalent
              dfs_vis.Set(i, true);
              dfs_group_id[i] = dfs_group_id[didx];
            } else {
              // for the same edge, switch the vertex order and try again
              std::swap(dfs_orders[i][0], dfs_orders[i][1]);
              if (TraversalPlanGenerator::SetTopoClosestOrder(
                      2, vertex_count, dfs_orders[i], dfs_intersect_indices,
                      conn, status) == vertex_count) {
                dfs_vis.Set(i, true);
                dfs_group_id[i] = dfs_group_id[didx];
              } else {
                // if fail, switch back to the normal vertex order
                std::swap(dfs_orders[i][0], dfs_orders[i][1]);
              }
            }
          }
        }
      }
    }
    group_dfs_ids.resize(cur_group_id);
    for (size_t i = 0; i < dfs_orders.size(); ++i) {
      group_dfs_ids[dfs_group_id[i]].push_back(i);
    }
  }

  // For a group of dfs ids, which are topo-equivalent,
  // change the order for each dfs based on the given candidate_order.
  // The candidate order is for the 1st dfs in this group
  static void AdjustDfsOrderInGroup(DfsIdGroup& dfs_ids,
                                    SearchSequence& candidate_order,
                                    std::vector<SearchSequence>& dfs_orders,
                                    bool rev_first_edge,
                                    const AllConnType& conn,
                                    size_t vertex_count) {
    if (rev_first_edge) {
      for (auto& dfs_id : dfs_ids) {
        std::swap(dfs_orders[dfs_id][0], dfs_orders[dfs_id][1]);
      }
    }

    // fix the 1st dfs
    size_t first_dfs_id = dfs_ids.at(0);
    dfs_orders[first_dfs_id].assign(candidate_order.begin(),
                                    candidate_order.end());

    AllConnType dfs_intersect_indices;
    TraversalPlanGenerateHelper::GetIndexBasedConnectivity(
        dfs_intersect_indices, conn, dfs_orders[first_dfs_id], vertex_count);

    // handle remaining dfs
    for (size_t gidx = 1; gidx < dfs_ids.size(); ++gidx) {
      size_t dfs_id = dfs_ids.at(gidx);
      BitSet status(vertex_count);
      status.Set(dfs_orders[dfs_id][0], true);
      status.Set(dfs_orders[dfs_id][1], true);
      auto reach_level = TraversalPlanGenerator::SetTopoClosestOrder(
          2, vertex_count, dfs_orders[dfs_id], dfs_intersect_indices, conn,
          status);
      assert(reach_level == vertex_count);
    }
  }

  // Given an order (in one group), represented by pattern_intersect_indices,
  // adjust another order (in another group) cur_seq,
  // so that the topology is close to the given order as much as possible.
  // For the remaining levels that cannot be equivalent, we choose the order
  // to achieve best pruning.
  // Return the number of levels that can set to topo-equivalent.
  static size_t AdjustTopoClosestOrder(
      SearchSequence& cur_seq, const AllConnType& pattern_intersect_indices,
      const AllConnType& conn, size_t vertex_count) {
    BitSet cur_status(vertex_count);
    cur_status.Set(cur_seq[0], true);
    cur_status.Set(cur_seq[1], true);
    auto reach_level = TraversalPlanGenerator::SetTopoClosestOrder(
        2, vertex_count, cur_seq, pattern_intersect_indices, conn, cur_status);

    // Since the levels [reach_level, vertex_count) cannot explore the
    // sharing, for the remaining levels, just choose the order that can
    // maximize the pruning
    if (reach_level < vertex_count) {
      cur_status.Reset(false);
      for (uintV i = 0; i < reach_level; ++i) {
        cur_status.Set(cur_seq[i], true);
      }
      TraversalPlanGenerator::GeneratePruneOrder(cur_seq, reach_level, conn,
                                                 vertex_count);
    }
    return reach_level;
  }

  // sort equi_group_dfs_ids based on the size of each group.
  static void SortGroupDfs(MultiDfsIdGroup& equi_group_dfs_ids) {
    size_t cur_group_num = equi_group_dfs_ids.size();

    // sort the groups so the one with more search sequences rank first
    // (group_num)^2
    for (size_t group_id = 0; group_id < cur_group_num; ++group_id) {
      for (size_t j = group_id + 1; j < cur_group_num; ++j) {
        if (equi_group_dfs_ids[group_id].size() <
            equi_group_dfs_ids[j].size()) {
          std::swap(equi_group_dfs_ids[group_id], equi_group_dfs_ids[j]);
        }
      }
    }
  }

  // Given the result of AdjustSharingOrder, i.e., dfs_orders,
  // for the prefix equivalence group for each level.
  // The result is stored in level_group_dfs_ids.
  // group_dfs_ids is all groups of topo-equivalent dfs.
  static void ConstructPrefixEquivalenceGroup(
      LevelMultiDfsIdGroup& level_group_dfs_ids,
      const std::vector<SearchSequence>& dfs_orders,
      const MultiDfsIdGroup& group_dfs_ids, const AllConnType& conn,
      size_t vertex_count) {
    // obtain the topo equivalence group, get as the copy
    MultiDfsIdGroup equi_group_dfs_ids(group_dfs_ids);
    SortGroupDfs(equi_group_dfs_ids);

    // construct level_group_dfs_ids
    level_group_dfs_ids.resize(vertex_count);
    // initialize first level
    level_group_dfs_ids[0].resize(1);
    for (size_t dfs_id = 0; dfs_id < dfs_orders.size(); ++dfs_id) {
      level_group_dfs_ids[0][0].push_back(dfs_id);
    }
    for (size_t l = 1; l < vertex_count; ++l) {
      MultiDfsIdGroup cur_group_dfs_ids;  // at level l, the group of
                                          // dfs ids that are equivalent
                                          // maintained so far

      for (size_t group_id = 0; group_id < equi_group_dfs_ids.size();
           ++group_id) {
        // try to merge with previous group,
        // from the group with larger weight first
        bool match_prev_group = false;
        for (size_t prev_group_id = 0; prev_group_id < cur_group_dfs_ids.size();
             ++prev_group_id) {
          size_t prev_first_dfs_id = cur_group_dfs_ids[prev_group_id][0];
          size_t equi_first_dfs_id = equi_group_dfs_ids[group_id][0];

          // for prev_first_dfs_id at level l, find the intersected
          // indices in the previsou levels
          BitSet prev_indices_status(vertex_count);
          TraversalPlanGenerateHelper::GetIndexBasedConnectivity(
              prev_indices_status, conn[dfs_orders[prev_first_dfs_id][l]],
              dfs_orders[prev_first_dfs_id], vertex_count);
          for (size_t i = l + 1; i < vertex_count; ++i) {
            prev_indices_status.Set(i, false);
          }

          // similar for equi_first_dfs_id at level l
          BitSet cur_indices_status(vertex_count);
          TraversalPlanGenerateHelper::GetIndexBasedConnectivity(
              cur_indices_status, conn[dfs_orders[equi_first_dfs_id][l]],
              dfs_orders[equi_first_dfs_id], vertex_count);
          for (size_t i = l + 1; i < vertex_count; ++i) {
            cur_indices_status.Set(i, false);
          }

          if (cur_indices_status.Equal(prev_indices_status)) {
            // For this level, connectivity for two groups are the same.
            // In order to merge into this group, also make sure
            // prefix equivalence at level l-1

            bool prev_prefix_equivalence = false;
            if (l == 1) {
              prev_prefix_equivalence = true;
            } else {
              MultiDfsIdGroup& last_group_dfs_ids = level_group_dfs_ids[l - 1];
              prev_prefix_equivalence = TraversalPlanGenerateHelper::SameGroup(
                  last_group_dfs_ids, prev_first_dfs_id, equi_first_dfs_id);
            }

            if (prev_prefix_equivalence) {
              match_prev_group = true;
              cur_group_dfs_ids[prev_group_id].insert(
                  cur_group_dfs_ids[prev_group_id].end(),
                  equi_group_dfs_ids[group_id].begin(),
                  equi_group_dfs_ids[group_id].end());
              break;
            }
          }
        }
        if (!match_prev_group) {
          // not match other groups at level l, create a new group
          cur_group_dfs_ids.push_back(equi_group_dfs_ids[group_id]);
        }
      }
      assert(cur_group_dfs_ids.size() > 0);
      level_group_dfs_ids[l].assign(cur_group_dfs_ids.begin(),
                                    cur_group_dfs_ids.end());
    }

    // sort dfs ids in each group in each level
    for (uintV l = 0; l < vertex_count; ++l) {
      for (auto& group : level_group_dfs_ids[l]) {
        std::sort(group.begin(), group.end());
      }
    }
  }

  // Used after calling GenerateTopoEquivalentOrder to have the search sequences
  // such that the pruning effect is maximized.
  // This function would adjust the ordering among multiple search sequences to
  // explore the sharing.
  // The adjusted result is stored in dfs_orders.
  // group_dfs_ids stores all groups of topo-equivalent dfs.
  static void AdjustSharingOrder(std::vector<SearchSequence>& dfs_orders,
                                 const MultiDfsIdGroup& group_dfs_ids,
                                 const AllConnType& conn, size_t vertex_count) {
    // obtain the topo equivalence group, get as the copy
    MultiDfsIdGroup equi_group_dfs_ids(group_dfs_ids);
    SortGroupDfs(equi_group_dfs_ids);

    size_t cur_group_num = equi_group_dfs_ids.size();

    // make adjustments for each group : try to merge with other groups
    for (size_t group_id = 1; group_id < cur_group_num; ++group_id) {
      // greedy: at each group_id, choose the order that achieves best sharing
      // with previous groups and merge with that group
      std::vector<SearchSequence> candidate_orders;
      std::vector<uintV> sharing_levels;
      std::vector<bool> rev_first_edge;

      // Try each previous group, larger weight comeing first,
      // to combine with this group, i.e, group_id
      for (size_t prev_group_id = 0; prev_group_id < group_id;
           ++prev_group_id) {
        size_t first_dfs_id = equi_group_dfs_ids[group_id][0];
        size_t prev_first_dfs_id = equi_group_dfs_ids[prev_group_id][0];

        // construct intersect_indices for prev_first_dfs_id
        AllConnType prev_intersect_indices;
        TraversalPlanGenerateHelper::GetIndexBasedConnectivity(
            prev_intersect_indices, conn, dfs_orders[prev_first_dfs_id],
            vertex_count);

        // try best to match prev_intersect_indices
        SearchSequence cur_seq(dfs_orders[first_dfs_id]);
        auto reach_level = AdjustTopoClosestOrder(
            cur_seq, prev_intersect_indices, conn, vertex_count);

        // try reverse the direction of first edge
        // maybe that can achieve better sharing
        SearchSequence cur_rev_seq(dfs_orders[first_dfs_id]);
        std::swap(cur_rev_seq[0], cur_rev_seq[1]);
        auto rev_reach_level = AdjustTopoClosestOrder(
            cur_rev_seq, prev_intersect_indices, conn, vertex_count);

        // Compare the sharing achieved, decide whether to
        // reverse the direction of the first edge
        if (reach_level >= rev_reach_level) {
          candidate_orders.push_back(cur_seq);
          sharing_levels.push_back(reach_level);
          rev_first_edge.push_back(false);
        } else {
          candidate_orders.push_back(cur_rev_seq);
          sharing_levels.push_back(rev_reach_level);
          rev_first_edge.push_back(true);
        }
      }

      // the dfs instances in group_id must be able to share in those levels
      // unshared with prev_group_id as they have the same topology partial
      // sharing with prev_group_id is a free lunch, so choose the one with
      // farthest sharing distance
      size_t best_order_candidate_idx = 0;
      size_t best_sharing_score = 0;
      for (size_t prev_group_id = 0; prev_group_id < group_id;
           ++prev_group_id) {
        if (sharing_levels[prev_group_id] > best_sharing_score) {
          best_sharing_score = sharing_levels[prev_group_id];
          best_order_candidate_idx = prev_group_id;
        }
      }
      // At least the first edge is shared
      assert(best_sharing_score > 0);

      AdjustDfsOrderInGroup(
          equi_group_dfs_ids[group_id],
          candidate_orders[best_order_candidate_idx], dfs_orders,
          rev_first_edge[best_order_candidate_idx], conn, vertex_count);
    }
  }
};

#endif
