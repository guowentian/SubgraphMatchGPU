#ifndef __HYBRID_PARTIAL_GROUP_CPU_COMPONENT_H__
#define __HYBRID_PARTIAL_GROUP_CPU_COMPONENT_H__

#include "HybridCPUComp.h"

class HybridPartialGroupCPUComponent : public HybridCPUComponent {
 public:
  HybridPartialGroupCPUComponent(TraversalPlan *plan,
                                 TrackPartitionedGraph *cpu_rel,
                                 bool materialize_result, size_t thread_num)
      : HybridCPUComponent(plan, cpu_rel, materialize_result, thread_num),
        plan_(plan) {
    paths_ = new uintV *[thread_num_];
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      paths_[thread_id] = new uintV[plan_->GetVertexCount()];
    }
  }
  ~HybridPartialGroupCPUComponent() {
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      delete[] paths_[thread_id];
      paths_[thread_id] = NULL;
    }
    delete[] paths_;
    paths_ = NULL;
  }

  virtual void ThreadExecute(size_t thread_id, long long &ans, uintV u,
                             uintV v) {
    paths_[thread_id][0] = u;
    paths_[thread_id][1] = v;

    if (CheckEquality(paths_[thread_id], 1, v)) return;

    size_t dfs_num = dfs_orders_.size();
    BitSet status(dfs_num);
    // check condition
    for (size_t dfs_id = 0; dfs_id < dfs_num; ++dfs_id) {
      bool val = CheckCondition(paths_[thread_id], v,
                                dfs_conditions_indices_[dfs_id][1]);
      status.Set(dfs_id, val);
    }
    if (status.Count() > 0) {
      PartialGroupDFS(thread_id, 2, paths_[thread_id], ans, status);
    }
  }

  virtual void Init() {
    HybridCPUComponent::Init();

    TraversalPlan *plan = static_cast<TraversalPlan *>(plan_);
    plan->LoadLevelMultiDfsIdGroup(level_group_dfs_ids_);
  }

 private:
  void PartialGroupDFS(size_t thread_id, size_t cur_level, uintV *path,
                       long long &ans, BitSet &status) {
    size_t cur_group_num = level_group_dfs_ids_[cur_level].size();
    for (size_t group_id = 0; group_id < cur_group_num; ++group_id) {
      // the set of dfs in the current group
      DfsIdGroup &group_dfs_ids = level_group_dfs_ids_[cur_level][group_id];
      bool valid_dfs_id = false;
      for (auto &dfs_id : group_dfs_ids) {
        if (status.Get(dfs_id)) {
          valid_dfs_id = true;
          break;
        }
      }
      // If no active dfs in the current group, skip this group
      if (!valid_dfs_id) {
        continue;
      }

      // obtain the connectivity in this group for this level
      size_t first_dfs_id = group_dfs_ids[0];
      std::vector<uintV> &intersect_indices =
          dfs_intersect_indices_[first_dfs_id][cur_level];

      std::vector<uintV> candidates;
      MWayIntersect<HOME_MADE>(path, cpu_relation_->GetRowPtrs(),
                               cpu_relation_->GetCols(), intersect_indices,
                               candidates);

      for (size_t i = 0; i < candidates.size(); ++i) {
        // non-equality
        if (CheckEquality(path, cur_level, candidates[i])) {
          continue;
        }

        // avoid double counting
        if (!CheckDuplicate(path, candidates[i], intersect_indices,
                            cpu_relation_->GetVertexPartitionMap(), 0, 1)) {
          continue;
        }

        // new_status tracks any active sequences for this group that survive
        // the constraint checking
        BitSet new_status(status.Size());
        for (auto &dfs_id : group_dfs_ids) {
          if (status.Get(dfs_id)) {
            // condition
            if (CheckCondition(path, candidates[i],
                               dfs_conditions_indices_[dfs_id][cur_level])) {
              new_status.Set(dfs_id, true);
            }
          }
        }

        if (cur_level + 1 == plan_->GetVertexCount()) {
          for (auto &dfs_id : group_dfs_ids) {
            if (new_status.Get(dfs_id)) {
              ++ans;
            }
          }
        } else {
          if (new_status.Count() > 0) {
            path[cur_level] = candidates[i];
            PartialGroupDFS(thread_id, cur_level + 1, path, ans, new_status);
          }
        }
      }
    }
  }

 protected:
  // (level, group_id, dfs_ids)
  LevelMultiDfsIdGroup level_group_dfs_ids_;

 private:
  TraversalPlan *plan_;
  uintV **paths_;
};

#endif
