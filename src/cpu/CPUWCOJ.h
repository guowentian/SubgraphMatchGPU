#ifndef __CPU_WORST_CASE_OPTIMAL_JOIN_H__
#define __CPU_WORST_CASE_OPTIMAL_JOIN_H__

#include <pthread.h>
#include "CPUFilter.h"
#include "CPUGraph.h"
#include "CPUIntersection.h"
#include "CPUPatternMatch.h"
#include "TimeMeasurer.h"

#if defined(OPENMP)
#include <omp.h>
#endif

class CPUWCOJoin : public CPUPatternMatch {
 public:
  CPUWCOJoin(TraversalPlan *plan, Graph *graph, size_t thread_num)
      : CPUPatternMatch(thread_num), plan_(plan), graph_(graph) {}

  virtual void Execute() {
    std::vector<std::vector<uintV> > intersect_levels;
    plan_->GetOrderedConnectivity(intersect_levels);
    AllCondType conditions;
    plan_->GetOrderedOrdering(conditions);

    omp_set_num_threads(thread_num_);

    TimeMeasurer timer;
    timer.StartTimer();

    long long total_match_count = 0;
    auto paths = new uintV *[thread_num_];
    for (size_t i = 0; i < thread_num_; ++i) {
      paths[i] = new uintV[plan_->GetVertexCount()];
    }

#pragma omp parallel for schedule(dynamic) reduction(+ : total_match_count)
    for (uintV u = 0; u < graph_->GetVertexCount(); ++u) {
      long long ans = 0;
      size_t thread_id = omp_get_thread_num();
#if defined(PROFILE)
      TimeMeasurer timer2;
      timer2.StartTimer();
#endif
      paths[thread_id][0] = u;
      DFS(thread_id, 1, paths[thread_id], ans, intersect_levels, conditions);
#if defined(PROFILE)
      timer2.EndTimer();
      thread_time_[thread_id] += timer2.GetElapsedMicroSeconds();
#endif
      total_match_count += ans;
    }

    for (size_t i = 0; i < thread_num_; ++i) {
      delete[] paths[i];
      paths[i] = NULL;
    }
    delete[] paths;
    paths = NULL;

    timer.EndTimer();

    this->SetTotalMatchCount(total_match_count);
    std::cout << "total_match_count=" << total_match_count
              << ", elapsed_time=" << timer.GetElapsedMicroSeconds() / 1000.0
              << "ms" << std::endl;
#if defined(PROFILE)
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      std::cout << "thread_id=" << thread_id
                << ",time=" << thread_time_[thread_id] / 1000.0 << "ms"
                << std::endl;
    }
#endif
  }

 private:
  void DFS(size_t thread_id, uintV cur_level, uintV *path, long long &ans,
           AllConnType &intersect_levels, AllCondType &conditions) {
    if (cur_level == plan_->GetVertexCount()) {
      ans++;
      return;
    }

    if (intersect_levels[cur_level].size() == 0) {
      for (uintV i = 0; i < graph_->GetVertexCount(); ++i) {
        if (CheckCondition(path, i, conditions[cur_level]) == false ||
            CheckEquality(path, cur_level, i))
          continue;
        path[cur_level] = i;
        DFS(thread_id, cur_level + 1, path, ans, intersect_levels, conditions);
      }
    } else {
      std::vector<uintV> candidates;
      MWayIntersect<HOME_MADE>(path, graph_->GetRowPtrs(), graph_->GetCols(),
                               intersect_levels[cur_level], candidates);

      for (size_t i = 0; i < candidates.size(); ++i) {
        if (CheckCondition(path, candidates[i], conditions[cur_level]) ==
                false ||
            CheckEquality(path, cur_level, candidates[i]))
          continue;
        path[cur_level] = candidates[i];
        DFS(thread_id, cur_level + 1, path, ans, intersect_levels, conditions);
      }
    }
  }

 private:
  TraversalPlan *plan_;
  Graph *graph_;
};

#endif
