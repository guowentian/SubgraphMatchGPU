#ifndef __QUERY_PLAN_REUSE_TRAVERSAL_PLAN_H__
#define __QUERY_PLAN_REUSE_TRAVERSAL_PLAN_H__

#include "ReuseTraversalCommon.h"
#include "ReuseTraversalOptimizer.h"
#include "ReuseTraversalPlanHardCode.h"
#include "TraversalPlan.h"

class ReuseTraversalPlan : public TraversalPlan {
 public:
  ReuseTraversalPlan(Query* query, size_t h_part_num, size_t dev_num,
                     ExecuteMode mode = HYBRID_CPU_GPU, Variant variant = O2,
                     bool autogen = false)
      : TraversalPlan(query, h_part_num, dev_num, mode, variant) {
    if (autogen) {
      ReuseTraversalOptimizer optimizer(query->GetQueryType(), vertex_count_,
                                        con_, order_);
      optimizer.GenerateOrder();
      seq_.assign(optimizer.seq_.begin(), optimizer.seq_.end());
      level_reuse_intersect_plan_.assign(optimizer.reuse_plan_.begin(),
                                         optimizer.reuse_plan_.end());
      for (size_t i = 0; i < level_reuse_intersect_plan_.size(); ++i) {
        level_reuse_intersect_plan_[i].TransformIndexBased(seq_);
      }

      this->GenConstraints();
    } else {
      ReuseTraversalPlanHardcode::Generate(seq_, level_reuse_intersect_plan_,
                                           query_);
      this->GenConstraints();
    }
  }
  ~ReuseTraversalPlan() {}

  virtual void Print() const {
    TraversalPlan::Print();
    std::cout << "------------- Reuse traversal plan for Q"
              << query_->GetQueryType() << "-----------------" << std::endl;
    for (size_t i = 0; i < vertex_count_; ++i) {
      std::cout << "level " << i << ", vertex=" << seq_[i] << ":";
      level_reuse_intersect_plan_[i].Print();
      std::cout << std::endl;
    }
    std::cout << "------------- end of Reuse traversal plan for Q"
              << query_->GetQueryType() << "-----------------" << std::endl;
  }

  LevelReuseIntersectPlan& GetLevelReuseIntersectPlan() {
    return level_reuse_intersect_plan_;
  }
  bool GetCacheRequired(uintV u) const { return cache_required_[u]; }

 protected:
  LevelReuseIntersectPlan level_reuse_intersect_plan_;
  std::vector<bool> cache_required_;

  // Reusable plan for inter-partition search sequences have not been supported
  // yet

  virtual void GenConstraints() {
    TraversalPlan::GenConstraints();

    cache_required_.resize(vertex_count_);
    for (uintV u = 0; u < vertex_count_; ++u) {
      bool cache_required = false;
      for (uintV u2 = u + 1; u2 < vertex_count_; ++u2) {
        for (auto& reuse_conn_meta :
             level_reuse_intersect_plan_[u2].GetReuseConnectivityMeta()) {
          if (reuse_conn_meta.source_vertex_ == u) {
            cache_required = true;
          }
        }
      }
      cache_required_[u] = cache_required;
    }
  }
};

#endif
