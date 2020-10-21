#ifndef __SAMPLE_MOTIF_FINDER_H__
#define __SAMPLE_MOTIF_FINDER_H__

#include <algorithm>
#include "MotifFinder.h"

class SampleMotifFinder : public MotifFinder {
 public:
  SampleMotifFinder(TraversalPlan* plan, Graph* rel, size_t thread_num)
      : MotifFinder(plan, rel, thread_num) {}
  ~SampleMotifFinder() {}
  virtual void Execute() {
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

    size_t pattern_vertex_count = plan_->GetVertexCount();
    auto path = new uintV*[thread_num_];
    auto permutate_orders = new uintV*[thread_num_];
    bool*** pred = new bool**[thread_num_];
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      path[thread_id] = new uintV[pattern_vertex_count];
      permutate_orders[thread_id] = new uintV[pattern_vertex_count];
      pred[thread_id] = new bool*[pattern_vertex_count];
      for (size_t i = 0; i < pattern_vertex_count; ++i) {
        pred[thread_id][i] = new bool[pattern_vertex_count];
      }
    }
    long long* thread_cross_times = new long long[thread_num_];
    memset(thread_cross_times, 0, sizeof(long long) * thread_num_);
    long long* thread_pass_vertices_num = new long long[thread_num_];
    memset(thread_pass_vertices_num, 0, sizeof(long long) * thread_num_);

    TimeMeasurer timer;
    timer.StartTimer();
#if defined(OPENMP)
    omp_set_num_threads(thread_num_);
    size_t levels_num = pattern_vertex_count;
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();
#pragma omp parallel for schedule(dynamic)
    for (uintV u = 0; u < graph_->GetVertexCount(); ++u) {
      for (size_t t = 0; t < kRandomWalkSampleNum; ++t) {
        // start from each vertex, sample kRandomWalkSampleNum times
        size_t thread_id = omp_get_thread_num();
        auto v = u;
        size_t pos = 0;
        while (1) {
          thread_pass_vertices_num[thread_id]++;
          path[thread_id][pos % levels_num] = v;
          pos++;
          if (pos >= levels_num) {
            // validate whether to form a motif
            for (size_t i = 0; i < levels_num; ++i)
              permutate_orders[thread_id][i] = i;
            for (size_t i = 0; i < levels_num; ++i) {
              for (size_t j = i + 1; j < levels_num; ++j) {
                auto v1 = path[thread_id][i];
                auto v2 = path[thread_id][j];
                auto posj = std::lower_bound(cols + row_ptrs[v1],
                                             cols + row_ptrs[v1 + 1], v2) -
                            cols;
                if (posj < row_ptrs[v1 + 1] && cols[posj] == v2) {
                  pred[thread_id][i][j] = pred[thread_id][j][i] = true;
                } else {
                  pred[thread_id][i][j] = pred[thread_id][j][i] = false;
                }
              }
            }
            do {
              if (MatchPattern(path[thread_id], permutate_orders[thread_id],
                               pred[thread_id], intersect_levels, conditions,
                               levels_num)) {
                UpdateCrossEdgeTimes(path[thread_id],
                                     permutate_orders[thread_id]);
                thread_cross_times[thread_id]++;
              }
            } while (std::next_permutation(
                permutate_orders[thread_id],
                permutate_orders[thread_id] + levels_num));
          }
          int rn = rand();
          double x = rn * 1.0 / RAND_MAX;
          if (x < ALPHA) break;
          // randomly choose a neighbor
          if (row_ptrs[v + 1] - row_ptrs[v] == 0) {
            v = u;
            continue;
          }
          size_t nindex = rn % (row_ptrs[v + 1] - row_ptrs[v]);
          auto nv = cols[row_ptrs[v] + nindex];
          v = nv;
        }
      }
    }
#endif
    long long total_cross_times = 0;
    long long total_pass_vertices_times = 0;
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      total_cross_times += thread_cross_times[thread_id];
      total_pass_vertices_times += thread_pass_vertices_num[thread_id];
    }
    timer.EndTimer();
    std::cout << "elapsed_time=" << timer.GetElapsedMicroSeconds() / 1000.0
              << "ms, total_cross_motif_times=" << total_cross_times
              << ",total_pass_vertices_times=" << total_pass_vertices_times
              << std::endl;
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      delete[] path[thread_id];
      path[thread_id] = NULL;
      delete[] permutate_orders[thread_id];
      permutate_orders[thread_id] = NULL;
    }
    delete[] path;
    path = NULL;
  }
  void UpdateCrossEdgeTimes(uintV* path, uintV* permutate_orders) {
    size_t pattern_vertex_count = plan_->GetVertexCount();
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();
    auto& connectivity = plan_->GetConnectivity();

    for (size_t l = 0; l < pattern_vertex_count; ++l) {
      for (size_t pred_id = 0; pred_id < connectivity[l].size(); ++pred_id) {
        size_t l2 = connectivity[l][pred_id];
        auto u = path[permutate_orders[l]];
        auto v = path[permutate_orders[l2]];
        auto pv =
            std::lower_bound(cols + row_ptrs[u], cols + row_ptrs[u + 1], v) -
            cols;
        assert(pv < row_ptrs[u + 1] && cols[pv] == v);
        locks_[pv].Lock();
        edge_cross_times_[pv]++;
        locks_[pv].Unlock();
      }
    }
  }
  static bool MatchPattern(uintV* path, uintV* permutate_orders, bool** pred,
                           AllConnType& intersect_levels,
                           AllCondType& conditions, size_t levels_num) {
    for (size_t l = 0; l < levels_num; ++l) {
      for (size_t j = 0; j < intersect_levels[l].size(); ++j) {
        size_t nl = intersect_levels[l][j];
        if (pred[permutate_orders[l]][permutate_orders[nl]] == false)
          return false;
      }
      auto u = path[permutate_orders[l]];
      for (size_t j = 0; j < conditions[l].size(); ++j) {
        size_t nl = conditions[l][j].second;
        auto v = path[permutate_orders[nl]];
        size_t condtype = conditions[l][j].first;
        if (condtype == LESS_THAN) {
          if (!(u < v)) return false;
        } else if (condtype == LARGER_THAN) {
          if (!(u > v)) return false;
        }
      }
      for (size_t nl = 0; nl < l; ++nl) {
        auto v = path[permutate_orders[nl]];
        if (u == v) return false;
      }
    }
    return true;
  }
};
#endif
