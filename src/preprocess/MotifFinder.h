#ifndef __MOTIF_FINDER_H__
#define __MOTIF_FINDER_H__

// discover motifs and store #times the motif cross for each edge
#include "CPUFilter.h"
#include "CPUGraph.h"
#include "CPUIntersection.h"
#include "TraversalPlan.h"

#include "SpinLock.h"
#include "TimeMeasurer.h"

#include <algorithm>

#if defined(OPENMP)
#include <omp.h>
#endif

class MotifFinder {
public:
  MotifFinder(TraversalPlan *plan, Graph *rel, size_t thread_num)
      : plan_(plan), graph_(rel), thread_num_(thread_num) {}
  ~MotifFinder() {}

  virtual void Execute() {
    // Assume the original vertex ids is the same as the search order
    AllConnType intersect_levels;
    AllCondType conditions;
    plan_->GetOrderedConnectivity(intersect_levels);
    plan_->GetOrderedOrdering(conditions);

    edge_cross_times_ = new size_t[graph_->GetEdgeCount()];
    memset(edge_cross_times_, 0, sizeof(size_t) * graph_->GetEdgeCount());
    locks_ = new SpinLock[graph_->GetEdgeCount()];
    for (size_t i = 0; i < graph_->GetEdgeCount(); ++i) {
      locks_[i].Init();
    }
#if defined(OPENMP)
    omp_set_num_threads(thread_num_);

    TimeMeasurer timer;
    timer.StartTimer();

    size_t pattern_vertex_count = plan_->GetVertexCount();
    total_match_count_ = 0;
    auto paths = new uintV *[thread_num_];
    for (size_t i = 0; i < thread_num_; ++i) {
      paths[i] = new uintV[pattern_vertex_count];
    }

    long long total_match_count = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : total_match_count)
    for (uintV u = 0; u < graph_->GetVertexCount(); ++u) {
      long long ans = 0;
      size_t thread_id = omp_get_thread_num();
      paths[thread_id][0] = u;
      DFS(thread_id, 1, paths[thread_id], ans, intersect_levels, conditions);
      // std::cout << "thread_id=" << thread_id << ",u=" << u << ",ans=" << ans
      // << std::endl;
      total_match_count_ += ans;
    }

    for (size_t i = 0; i < thread_num_; ++i) {
      delete[] paths[i];
      paths[i] = NULL;
    }
    delete[] paths;
    paths = NULL;

    timer.EndTimer();

    std::cout << "total_match_count_=" << total_match_count_
              << ", elapsed_time=" << timer.GetElapsedMicroSeconds() / 1000.0
              << "ms" << std::endl;
    total_match_count_ = total_match_count;
#endif // end of !OPENMP
  }

  void DFS(size_t thread_id, size_t cur_level, uintV *path, long long &ans,
           AllConnType &intersect_levels, AllCondType &conditions) {
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();
    size_t pattern_vertex_count = plan_->GetVertexCount();

    if (cur_level == pattern_vertex_count) {
      ans++;

      // for this instance, update the corresponding edge weights.
      auto &connectivity = plan_->GetConnectivity();
      for (size_t l = 0; l < pattern_vertex_count; ++l) {
        for (size_t pred_id = 0; pred_id < connectivity[l].size(); ++pred_id) {
          size_t l2 = connectivity[l][pred_id];
          auto u = path[l];
          auto v = path[l2];
          // edge u-v
          auto vindex =
              std::lower_bound(cols + row_ptrs[u], cols + row_ptrs[u + 1], v) -
              (cols + row_ptrs[u]) + row_ptrs[u];
          assert(vindex < row_ptrs[u + 1] && vindex >= row_ptrs[u]);
          locks_[vindex].Lock();
          edge_cross_times_[vindex]++;
          locks_[vindex].Unlock();
        }
      }
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
      std::vector<uintV> res[2];
      for (size_t j = 0; j < intersect_levels[cur_level].size(); ++j) {
        size_t p2 = intersect_levels[cur_level][j];
        auto first = path[p2];
        auto first_begin = &cols[row_ptrs[first]];
        auto first_end = &cols[row_ptrs[first + 1]];

        if (j == 0) {
          res[j % 2].assign(first_begin, first_end);
        } else {
          size_t max_size = std::min((size_t)(first_end - first_begin),
                                     res[(j + 1) % 2].size());
          res[j % 2].resize(max_size);
          size_t res_size = SortedIntersection(
              first_begin, first_end, res[(j + 1) % 2].begin(),
              res[(j + 1) % 2].end(), res[j % 2].begin());
          assert(res_size <= max_size);
          res[j % 2].resize(res_size);
        }
      }

      // based on the candidates set
      std::vector<uintV> &candidates =
          res[(intersect_levels[cur_level].size() + 1) % 2];
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
  size_t *GetEdgeCrossTimes() const { return edge_cross_times_; }
  size_t GetTotalMatchCount() const { return total_match_count_; }

public:
  TraversalPlan *plan_;
  Graph *graph_;
  size_t thread_num_;

  size_t total_match_count_;
  size_t *edge_cross_times_;
  SpinLock *locks_;
};

#endif
