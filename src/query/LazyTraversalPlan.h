#ifndef __LAZY_MATERIALIZATION_TRAVERSAL_PLAN_H__
#define __LAZY_MATERIALIZATION_TRAVERSAL_PLAN_H__

#include "LazyTraversalCommon.h"
#include "TraversalPlan.h"

#include "LazyTraversalPlanGenerator.h"
#include "LazyTraversalPlanHardCode.h"
#include "LazyTraversalPlanUtils.h"

class LazyTraversalPlan : public TraversalPlan {
 public:
  LazyTraversalPlan(
      Query* query, size_t h_part_num, size_t dev_num,
      ExecuteMode mode = HYBRID_CPU_GPU, Variant variant = O2,
      LazyTraversalCompressLevel compress_level =
          LazyTraversalCompressLevel::COMPRESS_LEVEL_NON_MATERIALIZE_OPT)
      : TraversalPlan(query, h_part_num, dev_num, mode, variant) {
    if (compress_level == LazyTraversalCompressLevel::COMPRESS_LEVEL_SPECIAL) {
      intra_plan_compress_level_ =
          LazyTraversalCompressLevel::COMPRESS_LEVEL_SPECIAL;
      inter_plan_compress_level_ =
          LazyTraversalCompressLevel::COMPRESS_LEVEL_NON_MATERIALIZE_OPT;
    } else {
      intra_plan_compress_level_ = inter_plan_compress_level_ = compress_level;
    }
    LazyTraversalPlanHardcode::GenerateSearchSequence(seq_, query_);
    GenConstraints();
  }

  virtual void OptimizePlan() {
    TraversalPlan::OptimizePlan();

    if (execute_mode_ == HYBRID_CPU_GPU) {
      auto& dfs_orders = this->GetSearchSequences();
      auto& group_dfs_ids = this->GetGroupDfsIds();
      size_t dfs_num = dfs_orders.size();
      size_t group_num = group_dfs_ids.size();

      inter_part_exec_seq_.resize(dfs_num);
      inter_part_materialized_vertices_.resize(dfs_num);
      inter_part_computed_unmaterialized_vertices_.resize(dfs_num);

      for (size_t group_id = 0; group_id < group_num; ++group_id) {
        size_t first_dfs_id = group_dfs_ids[group_id][0];
        std::vector<LazyTraversalEntry> exec_seq;

        LazyTraversalPlanGenerator::GenerateExecuteSequence(
            exec_seq, dfs_orders[first_dfs_id], con_, vertex_count_,
            inter_plan_compress_level_, true);

        LazyTraversalPlanUtils::GetIndexBasedExecuteSequence(
            exec_seq, dfs_orders[first_dfs_id], vertex_count_);

        // all dfs in the same group share the same exec_seq,
        // which is index-based
        for (auto dfs_id : group_dfs_ids[group_id]) {
          inter_part_exec_seq_[dfs_id].assign(exec_seq.begin(), exec_seq.end());
          LazyTraversalPlanUtils::GenerateConstraints(
              inter_part_materialized_vertices_[dfs_id],
              inter_part_computed_unmaterialized_vertices_[dfs_id],
              inter_part_exec_seq_[dfs_id], vertex_count_);
        }
      }
    }
  }

  virtual void Print() const {
    TraversalPlan::Print();
    std::cout << "--------------------- Lazy traversal plan for Q"
              << query_->GetQueryType() << "--------------------" << std::endl;

    PrintLazyTraversalPlanData(exec_seq_, materialized_vertices_,
                               computed_unmaterialized_vertices_);
    size_t dfs_num = dfs_orders_.size();
    for (size_t dfs_id = 0; dfs_id < dfs_num; ++dfs_id) {
      std::cout << "dfs_id=" << dfs_id << ":";
      PrintLazyTraversalPlanData(
          inter_part_exec_seq_[dfs_id],
          inter_part_materialized_vertices_[dfs_id],
          inter_part_computed_unmaterialized_vertices_[dfs_id]);
    }

    std::cout << "--------------------- end of Lazy traversal plan for Q"
              << query_->GetQueryType() << "--------------------" << std::endl;
  }

  //////////////////// operations for intra-partition  ////////////
  //
  // the ordering constraints enforced for COMPUTE step,
  // i.e., the ordering for those vertices that have been materialized
  // before this vertex is computed.
  void GetComputeCondition(AllCondType& ret) {
    LazyTraversalPlanUtils::GetComputeCondition(ret, exec_seq_, order_,
                                                vertex_count_);
  }

  // The ordering constraints enforced for MATERIALIZE step.
  // The ordering for those vertices that are materialized after
  // this vertex is computed and before this vertex is materialized.
  void GetMaterializeCondition(AllCondType& ret) {
    LazyTraversalPlanUtils::GetMaterializeCondition(ret, exec_seq_, order_,
                                                    vertex_count_);
  }

  // The ordering constraints used for FILTER_COMPUTE step.
  // The ordering for those vertices that are materialized after
  // the last time when this vertex is computed
  // and before this vertex is filter-computed
  void GetFilterCondition(AllCondType& ret) {
    LazyTraversalPlanUtils::GetFilterCondition(ret, exec_seq_, order_,
                                               vertex_count_);
  }

  size_t GetOperationIndex(LazyTraversalOperation op, uintV u) const {
    return LazyTraversalPlanUtils::GetOperationIndex(exec_seq_, op, u);
  }

  size_t GetCountOperationIndex() const {
    for (size_t i = 0; i < exec_seq_.size(); ++i) {
      if (exec_seq_[i].first == COUNT || exec_seq_[i].first == COMPUTE_COUNT) {
        return i;
      }
    }
    return exec_seq_.size();
  }

  // ==================  getter ==================
  std::vector<LazyTraversalEntry>& GetExecuteOperations() { return exec_seq_; }
  MultiVTGroup& GetMaterializedVertices() { return materialized_vertices_; }
  MultiVTGroup& GetComputedUnmaterializedVertices() {
    return computed_unmaterialized_vertices_;
  }

  std::vector<std::vector<LazyTraversalEntry>>&
  GetInterPartitionExecuteOperations() {
    return inter_part_exec_seq_;
  }
  std::vector<MultiVTGroup>& GetInterPartitionMaterializedVertices() {
    return inter_part_materialized_vertices_;
  }
  std::vector<MultiVTGroup>& GetInterPartitionComputedUnmaterializedVertices() {
    return inter_part_computed_unmaterialized_vertices_;
  }
  LazyTraversalCompressLevel GetIntraPartitionPlanCompressLevel() const {
    return intra_plan_compress_level_;
  }

 protected:
  std::vector<LazyTraversalEntry> exec_seq_;
  // for each level l in exec_seq_, the vertices that have been
  // materialized before l
  MultiVTGroup materialized_vertices_;
  // for each level l in exec_seq_, the vertices that have been
  // computed but not materialized.
  MultiVTGroup computed_unmaterialized_vertices_;

  // To control the implementation of counting instances.
  // Either use the fastest version that is hardcoded,
  // or use the general version that is slow but can support any general pattern
  LazyTraversalCompressLevel intra_plan_compress_level_;
  LazyTraversalCompressLevel inter_plan_compress_level_;

  ////////////////  inter partition data structures //////////////

  // note that the inter-partition plan data is based on the index
  // instead of the original vertex ids.
  std::vector<std::vector<LazyTraversalEntry>> inter_part_exec_seq_;
  std::vector<MultiVTGroup> inter_part_materialized_vertices_;
  std::vector<MultiVTGroup> inter_part_computed_unmaterialized_vertices_;

 protected:
  virtual void GenConstraints() {
    QueryType query_type = query_->GetQueryType();
    if (intra_plan_compress_level_ == COMPRESS_LEVEL_SPECIAL) {
      // if the intra-partition plan is allowed for hardcode,
      if (query_type == Q2 || query_type == Q5 || query_type == Q6) {
        // the query type is suitable for hardcode
        LazyTraversalPlanGenerator::SpecialExecuteSequence(
            exec_seq_, query_type, seq_, con_, vertex_count_, false);
      } else {
        // but the query type is not suitable for hardcode,
        // still fall back to EXT_NON_MATERIALIZE
        LazyTraversalPlanGenerator::GenerateExecuteSequence(
            exec_seq_, seq_, con_, vertex_count_,
            COMPRESS_LEVEL_NON_MATERIALIZE_OPT, false);
      }
    } else {
      LazyTraversalPlanGenerator::GenerateExecuteSequence(
          exec_seq_, seq_, con_, vertex_count_, intra_plan_compress_level_,
          false);
    }

    LazyTraversalPlanUtils::GenerateConstraints(
        materialized_vertices_, computed_unmaterialized_vertices_, exec_seq_,
        vertex_count_);
  }
};

#endif
