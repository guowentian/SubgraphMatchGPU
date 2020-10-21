#ifndef __HYBRID_CPU_COMPONENT_H__
#define __HYBRID_CPU_COMPONENT_H__

#include <cstring>
#include <vector>
#include "CPUFilter.h"
#include "CPUGraph.h"
#include "CPUIntersection.h"
#include "CountProfiler.h"
#include "PhaseProfiler.h"
#include "Task.h"
#include "TimeMeasurer.h"
#include "TraversalPlan.h"

// vanilla version, cpu threads process inter-partition instances without
// optimization
class HybridCPUComponent {
 public:
  HybridCPUComponent(TraversalPlan *plan, TrackPartitionedGraph *cpu_rel,
                     bool materialize_result, size_t thread_num)
      : plan_(plan), cpu_relation_(cpu_rel), thread_num_(thread_num) {
    paths_ = new uintV *[thread_num_];
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      paths_[thread_id] = new uintV[plan_->GetVertexCount()];
    }
    phase_profiler_ = new PhaseProfiler(thread_num_);
    count_profiler_ = new CountProfiler(thread_num_);
  }

  ~HybridCPUComponent() {
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      delete[] paths_[thread_id];
      paths_[thread_id] = NULL;
    }
    delete[] paths_;
    paths_ = NULL;
    delete phase_profiler_;
    phase_profiler_ = NULL;
    delete count_profiler_;
    count_profiler_ = NULL;
  }

  void Execute(Task *task) {
    assert(task->task_type_ == INTER_PARTITION);
    InterPartTask *inter_part_task = static_cast<InterPartTask *>(task);
    long long &ans = *(inter_part_task->ans_);

    long long *thread_ans = new long long[thread_num_];
    memset(thread_ans, 0, sizeof(long long) * thread_num_);
#pragma omp parallel for num_threads(thread_num_)
    for (uintE e = inter_part_task->start_offset_;
         e < inter_part_task->end_offset_; ++e) {
      uintV v = cpu_relation_->GetInterCols()[e];
      uintE *off_array = cpu_relation_->GetInterRowPtrs();
      size_t n = cpu_relation_->GetVertexCount();
      uintV u = std::upper_bound(off_array, off_array + n + 1, e) - off_array;
      u--;

      size_t thread_id = omp_get_thread_num();
      this->ThreadExecute(thread_id, thread_ans[thread_id], u, v);
    }

    for (size_t i = 0; i < thread_num_; ++i) {
      ans += thread_ans[i];
    }
    delete thread_ans;
    thread_ans = NULL;
  }

  void ThreadExecute(Task *task) {
    InterPartTask *inter_part_task = static_cast<InterPartTask *>(task);
    long long &ans = *(inter_part_task->ans_);

    for (uintE e = inter_part_task->start_offset_;
         e < inter_part_task->end_offset_; ++e) {
      uintV v = cpu_relation_->GetInterCols()[e];
      uintE *off_array = cpu_relation_->GetInterRowPtrs();
      size_t n = cpu_relation_->GetVertexCount();
      uintV u = std::upper_bound(off_array, off_array + n + 1, e) - off_array;
      u--;

      size_t thread_id = omp_get_thread_num();
      this->ThreadExecute(thread_id, ans, u, v);
    }
  }

  virtual void ThreadExecute(size_t thread_id, long long &ans, uintV u,
                             uintV v) {
    // given a starting data edge (v0,v1)
    paths_[thread_id][0] = u;
    paths_[thread_id][1] = v;
    // plain dfs search for inter-partition instances:
    // handle each search sequence one by one without sharing
    for (size_t dfs_id = 0; dfs_id < dfs_orders_.size(); ++dfs_id) {
      if (u == v || !CheckCondition(paths_[thread_id], v,
                                    dfs_conditions_indices_[dfs_id][1]))
        continue;
      DFS(thread_id, 2, paths_[thread_id], ans, dfs_intersect_indices_[dfs_id],
          dfs_conditions_indices_[dfs_id]);
    }
  }

  virtual void Init() {
    plan_->LoadSearchSequences(dfs_orders_);
    plan_->GetOrderedIndexBasedConnectivity(dfs_intersect_indices_);
    plan_->GetOrderedIndexBasedOrdering(dfs_conditions_indices_);
  }

  virtual void ReportProfile() {
    phase_profiler_->Report();
    count_profiler_->Report();
  }

 private:
  void DFS(size_t thread_id, size_t match_vertices_count, uintV *path,
           long long &ans, AllConnType &intersect_levels,
           AllCondType &conditions) {
    if (match_vertices_count == plan_->GetVertexCount()) {
      ans++;
      return;
    }
    assert(intersect_levels[match_vertices_count].size() > 0);

    std::vector<uintV> candidates;
    MWayIntersect<HOME_MADE>(
        path, cpu_relation_->GetRowPtrs(), cpu_relation_->GetCols(),
        intersect_levels[match_vertices_count], candidates);

    for (size_t i = 0; i < candidates.size(); ++i) {
      if (!CheckDuplicate(path, candidates[i],
                          intersect_levels[match_vertices_count],
                          cpu_relation_->GetVertexPartitionMap(), 0, 1)) {
        continue;
      }

      if (!CheckCondition(path, candidates[i],
                          conditions[match_vertices_count]) ||
          CheckEquality(path, match_vertices_count, candidates[i])) {
        continue;
      }

      path[match_vertices_count] = candidates[i];
      DFS(thread_id, match_vertices_count + 1, path, ans, intersect_levels,
          conditions);
    }
  }

 protected:
  TrackPartitionedGraph *cpu_relation_;
  size_t thread_num_;

  // search sequence for each dfs
  std::vector<SearchSequence> dfs_orders_;
  // for each dfs, the ordered index based connectivity
  std::vector<AllConnType> dfs_intersect_indices_;
  // for each dfs, the ordered index based ordering
  std::vector<AllCondType> dfs_conditions_indices_;

  bool materialize_result_;

  CountProfiler *count_profiler_;
  PhaseProfiler *phase_profiler_;

 private:
  TraversalPlan *plan_;
  // store partial instances
  uintV **paths_;
};

#endif
