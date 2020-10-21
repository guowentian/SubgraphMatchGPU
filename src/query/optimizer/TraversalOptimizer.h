#ifndef __TRAVERSAL_OPTIMIZER_H__
#define __TRAVERSAL_OPTIMIZER_H__

#include "MultiTraversalPlanGenerator.h"
#include "TraversalPlanGenerator.h"

class TraversalOptimizer : public Optimizer {
 public:
  TraversalOptimizer(QueryType query_type, size_t vertex_count,
                     AllConnType& con, AllCondType& order)
      : Optimizer(query_type, vertex_count, con, order) {}

  virtual void GenerateOrder() {
    TraversalPlanGenerator::GeneratePruneOrder(seq_, 0, con_, vertex_count_);
  }

  // select the order for multiple search seequences in
  // the inter-partition search
  virtual void GenerateInterPartitionOrder() {
    MultiTraversalPlanGenerator::GenerateTopoEquivalentOrder(
        dfs_orders_, group_dfs_ids_, con_, vertex_count_);
    MultiTraversalPlanGenerator::AdjustSharingOrder(dfs_orders_, group_dfs_ids_,
                                                    con_, vertex_count_);
    MultiTraversalPlanGenerator::ConstructPrefixEquivalenceGroup(
        level_group_dfs_ids_, dfs_orders_, group_dfs_ids_, con_, vertex_count_);
  }

  // after GenerateTopoEquivalentOrder(), instead of
  // AdjustSharingOrder(), directly set level_group_dfs_ids_.
  // This is a weaker level of sharing compared with AdjustSharingOrder()
  void SetTopoEquivalentLevelGroupDfsIds() {
    level_group_dfs_ids_.resize(vertex_count_);
    for (uintV l = 0; l < vertex_count_; ++l) {
      level_group_dfs_ids_[l].assign(group_dfs_ids_.begin(),
                                     group_dfs_ids_.end());
    }
  }

  virtual double GetCost() const {
    size_t ret = 0;
    std::vector<double> cur_group_size(level_group_dfs_ids_[1].size(), 1.0);
    ret += level_group_dfs_ids_[1].size() * 1.0;
    for (size_t level = 2; level < vertex_count_; ++level) {
      std::vector<double> nxt_group_size(level_group_dfs_ids_[level].size(),
                                         0.0);
      for (size_t group_id = 0; group_id < level_group_dfs_ids_[level].size();
           ++group_id) {
        size_t first_dfs_id = level_group_dfs_ids_[level][group_id][0];
        size_t prev_level = level - 1;
        size_t prev_group_id = 0;
        for (prev_group_id = 0;
             prev_group_id < level_group_dfs_ids_[prev_level].size();
             ++prev_group_id) {
          bool found = false;
          for (size_t j = 0;
               j < level_group_dfs_ids_[prev_level][prev_group_id].size();
               ++j) {
            if (level_group_dfs_ids_[prev_level][prev_group_id][j] ==
                first_dfs_id) {
              found = true;
            }
          }
          if (found) break;
        }
        // group_id is derived from prev_group_id

        size_t incident_num = 0;
        for (size_t pred_id = 0;
             pred_id < con_[dfs_orders_[first_dfs_id][level]].size();
             ++pred_id) {
          size_t plevel = con_[dfs_orders_[first_dfs_id][level]][pred_id];
          bool visited = false;
          for (size_t j = 0; j < level; ++j) {
            if (dfs_orders_[first_dfs_id][j] == plevel) {
              visited = true;
            }
          }
          if (visited) {
            incident_num++;
          }
        }
        // incident_num is #connecitivty to previous levels

        double cur_size = cur_group_size[prev_group_id];
        // #candidates in this level=(cur_size*EXPAND_FACTOR)
        ret += (cur_size * EXPAND_FACTOR) * incident_num;

        double nxt_size = cur_size;
        nxt_size *= EXPAND_FACTOR;
        for (size_t i = 1; i < incident_num; ++i) {
          nxt_size *= PRUNE_FACTOR;
        }
        nxt_group_size[group_id] = nxt_size;
      }
      cur_group_size.swap(nxt_group_size);
    }
    return ret;
  }

  virtual void Print() const {
    std::cout << "traversal optimizer, dfs orders: " << dfs_orders_.size()
              << std::endl;
    std::cout << "intra-partition order:";
    for (auto u : seq_) std::cout << " " << u;
    std::cout << std::endl;

    for (size_t j = 0; j < dfs_orders_.size(); ++j) {
      std::cout << j << ":";
      for (uintV l = 0; l < vertex_count_; ++l) {
        std::cout << " " << dfs_orders_[j][l];
      }
      std::cout << std::endl;
    }
    for (size_t i = 0; i < group_dfs_ids_.size(); ++i) {
      std::cout << "group " << i << ":";
      for (size_t j = 0; j < group_dfs_ids_[i].size(); ++j) {
        std::cout << " " << group_dfs_ids_[i][j];
      }
      std::cout << std::endl;
    }

    for (uintV l = 0; l < vertex_count_; ++l) {
      for (size_t group_id = 0; group_id < level_group_dfs_ids_[l].size();
           ++group_id) {
        std::cout << "(level=" << l << ",group_id=" << group_id << "):";
        for (size_t gidx = 0; gidx < level_group_dfs_ids_[l][group_id].size();
             ++gidx) {
          std::cout << " " << level_group_dfs_ids_[l][group_id][gidx];
        }
        std::cout << std::endl;
      }
    }
  }

 public:
  // the match order for intra-partition search.
  SearchSequence seq_;

  // The results generated by the optimizer.
  std::vector<SearchSequence> dfs_orders_;  // the search sequence for each dfs

  // The result generated by topo-equivalence order,
  // which is the input for AdjustSharingOrder().
  // Several groups of dfs ids. In each group, the dfs are
  // topo equivalent
  MultiDfsIdGroup group_dfs_ids_;

  // For each level, there are multiple groups of dfs ids.
  // The dfs in the same group are prefix equivalent to
  // each other at the current level
  LevelMultiDfsIdGroup level_group_dfs_ids_;
};

#endif
