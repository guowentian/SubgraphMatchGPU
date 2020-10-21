#ifndef __TRAVERSAL_PLAN_H__
#define __TRAVERSAL_PLAN_H__

#include <vector>
#include "Plan.h"
#include "TraversalOptimizer.h"

class TraversalPlan : public Plan {
 public:
  TraversalPlan(Query* query, size_t h_part_num, size_t dev_num,
                ExecuteMode mode = HYBRID_CPU_GPU, Variant variant = O2)
      : Plan(query, h_part_num, dev_num, mode, variant) {
    // The order of intra-partition search sequences is pre-assigned and the
    // its data structures are generated in the constructor.
    // Ideally they should be generated in OptimizePlan()
    // and the time can be included for evaluation.
    // But this is fast and can be neligible.
    GenSearchSequence();
    GenConstraints();
  }
  ~TraversalPlan() {}

  virtual void OptimizePlan() {
    if (execute_mode_ == HYBRID_CPU_GPU) {
      GenInterPartitionSharingPlan();
    }
  }

  virtual void Print() const {
    std::cout << "----- traversal query plan for Q" << query_->GetQueryType()
              << "-----------" << std::endl;
    for (uintV u = 0; u < vertex_count_; ++u) {
      std::cout << u << ":";
      for (size_t j = 0; j < con_[u].size(); ++j) {
        std::cout << " " << con_[u][j];
      }
      std::cout << std::endl;
    }

    std::cout << "ordering:";
    for (uintV u = 0; u < vertex_count_; ++u) {
      for (size_t j = 0; j < order_[u].size(); ++j) {
        std::cout << " (" << u;
        if (order_[u][j].first == LESS_THAN) {
          std::cout << "<";
        } else if (order_[u][j].first == LARGER_THAN) {
          std::cout << ">";
        }
        std::cout << order_[u][j].second << ")";
      }
    }
    std::cout << std::endl;

    std::cout << "search sequence for intra-partition:";
    for (auto& v : seq_) {
      std::cout << " " << v;
    }
    std::cout << std::endl;

    std::cout << "search sequence for inter-partition:" << std::endl;
    for (size_t dfs_id = 0; dfs_id < dfs_orders_.size(); ++dfs_id) {
      std::cout << "dfs_id=" << dfs_id << ":";
      for (auto& v : dfs_orders_[dfs_id]) {
        std::cout << " " << v;
      }
      std::cout << std::endl;
    }

    /*for (size_t group_id = 0; group_id < group_dfs_ids_.size(); ++group_id) {
      std::cout << "group_id=" << group_id << ":";
      for (auto& dfs_id : group_dfs_ids_[group_id]) {
        std::cout << " " << dfs_id;
      }
      std::cout << std::endl;
    }*/

    for (size_t level = 0; level < level_group_dfs_ids_.size(); ++level) {
      std::cout << "level=" << level << std::endl;
      for (size_t group_id = 0; group_id < level_group_dfs_ids_[level].size();
           ++group_id) {
        std::cout << "group_id=" << group_id << ":";
        for (auto& dfs_id : level_group_dfs_ids_[level][group_id]) {
          std::cout << " " << dfs_id;
        }
        std::cout << std::endl;
      }
    }
    std::cout << "------------- end of traversal query plan for Q"
              << query_->GetQueryType() << "------------" << std::endl;
  }

  ////// =============  getter ================== //////////////////

  /// ==== intra-partition ====
  void GetOrderedConnectivity(AllConnType& ret) const {
    Plan::GetOrderedConnectivity(ret, con_);
  }
  void GetOrderedOrdering(AllCondType& ret) const {
    Plan::GetOrderedOrdering(ret, order_);
  }
  void GetWholeOrderedOrdering(AllCondType& ret) const {
    Plan::GetWholeOrderedOrdering(ret, order_);
  }

  //// ===== inter-partition ====
  void GetOrderedIndexBasedConnectivity(std::vector<AllConnType>& ret) const {
    size_t dfs_num = dfs_orders_.size();
    ret.resize(dfs_num);
    for (size_t dfs_id = 0; dfs_id < dfs_num; ++dfs_id) {
      Plan::GetOrderedIndexBasedConnectivity(ret[dfs_id], con_,
                                             dfs_orders_[dfs_id]);
    }
  }
  void GetOrderedIndexBasedOrdering(std::vector<AllCondType>& ret) const {
    size_t dfs_num = dfs_orders_.size();
    ret.resize(dfs_num);
    for (size_t dfs_id = 0; dfs_id < dfs_num; ++dfs_id) {
      Plan::GetOrderedIndexBasedOrdering(ret[dfs_id], order_,
                                         dfs_orders_[dfs_id]);
    }
  }
  void GetWholeOrderedIndexBasedOrdering(std::vector<AllCondType>& ret) const {
    size_t dfs_num = dfs_orders_.size();
    ret.resize(dfs_num);
    for (size_t dfs_id = 0; dfs_id < dfs_num; ++dfs_id) {
      Plan::GetWholeOrderedIndexBasedOrdering(ret[dfs_id], order_,
                                              dfs_orders_[dfs_id]);
    }
  }

  //// ==== get each specific member ====
  size_t GetVertexCount() const { return vertex_count_; }
  AllConnType& GetConnectivity() { return con_; }
  AllCondType& GetOrdering() { return order_; }
  SearchSequence& GetSearchSequence() { return seq_; }

  std::vector<SearchSequence>& GetSearchSequences() { return dfs_orders_; }
  MultiDfsIdGroup& GetGroupDfsIds() { return group_dfs_ids_; }
  LevelMultiDfsIdGroup& GetLevelGroupDfsIds() { return level_group_dfs_ids_; }

  void LoadSearchSequences(std::vector<SearchSequence>& obj) {
    obj.assign(dfs_orders_.begin(), dfs_orders_.end());
  }
  void LoadMultiDfsIdGroup(MultiDfsIdGroup& obj) {
    obj.assign(group_dfs_ids_.begin(), group_dfs_ids_.end());
  }
  void LoadLevelMultiDfsIdGroup(LevelMultiDfsIdGroup& obj) {
    obj.assign(level_group_dfs_ids_.begin(), level_group_dfs_ids_.end());
  }

 protected:
  SearchSequence seq_;  // search sequence
  // Based on the search sequence, relabel the vertex id,
  // then have the new connectivity and ordering constraint for the pattern
  // During the search process, we use the newly labeled vertex ids
  size_t vertex_count_;
  AllConnType con_;
  AllCondType order_;

  // ========= for inter-partition workload =========
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

  void AssignSharingPlanResult(TraversalOptimizer* optimizer) {
    // the inter-partition cpu component would use dfs_orders_
    // and level_group_dfs_ids_ in execution
    dfs_orders_.assign(optimizer->dfs_orders_.begin(),
                       optimizer->dfs_orders_.end());
    group_dfs_ids_.assign(optimizer->group_dfs_ids_.begin(),
                          optimizer->group_dfs_ids_.end());
    level_group_dfs_ids_.assign(optimizer->level_group_dfs_ids_.begin(),
                                optimizer->level_group_dfs_ids_.end());
  }
  void GenInterPartitionSharingPlan() {
    TraversalOptimizer* optimizer = new TraversalOptimizer(
        query_->GetQueryType(), vertex_count_, con_, order_);
    optimizer->GenerateInterPartitionOrder();
    AssignSharingPlanResult(optimizer);
    delete optimizer;
    optimizer = NULL;
  }

  void GenSearchSequence() {
    // Currently we hardcode a good search sequence specified in Query
    // and use this sequence in the search.
    // By right, this should be generated by the optimizer with a selective
    // order strategy to maximize the pruning effect and reduce the intermediate
    // result.
    for (uintV u = 0; u < query_->GetVertexCount(); ++u) {
      seq_.push_back(u);
    }
  }
  virtual void GenConstraints() {
    // based on seq_
    vertex_count_ = query_->GetVertexCount();
    AllConnType& ocon = query_->GetConnectivity();
    AllCondType& oorder = query_->GetOrdering();
    SearchSequence otn_map(vertex_count_);  // the map from old to new
    for (uintV u = 0; u < vertex_count_; ++u) {
      uintV ou = seq_[u];
      otn_map[ou] = u;
    }

    // init, as con_ and order_ may be constructed in the previous call
    // of GenConstraints
    con_.clear();
    order_.clear();
    con_.resize(vertex_count_);
    order_.resize(vertex_count_);
    for (uintV u = 0; u < vertex_count_; ++u) {
      uintV ou = seq_[u];
      con_[u].assign(ocon[ou].begin(), ocon[ou].end());
      for (size_t j = 0; j < con_[u].size(); ++j) {
        uintV ov = con_[u][j];
        con_[u][j] = otn_map[ov];
      }
      order_[u].assign(oorder[ou].begin(), oorder[ou].end());
      for (size_t j = 0; j < order_[u].size(); ++j) {
        uintV ov = order_[u][j].second;
        order_[u][j].second = otn_map[ov];
      }
    }
  }
};

#endif
