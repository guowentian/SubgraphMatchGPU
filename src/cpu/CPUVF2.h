#ifndef __CPU_VF2_H__
#define __CPU_VF2_H__

#include "CPUFilter.h"
#include "CPUGraph.h"
#include "CPUIntersection.h"
#include "CPUPatternMatch.h"
#include "TimeMeasurer.h"

#if defined(OPENMP)
#include <omp.h>
#endif
#include <unordered_map>

class CPUVF2 : public CPUPatternMatch {
  public:
  CPUVF2(TraversalPlan *plan, Graph *graph, size_t thread_num)
      : CPUPatternMatch(thread_num), plan_(plan), graph_(graph) {}
  virtual ~CPUVF2() {}

  virtual void Execute() {
    std::vector<std::vector<uintV> > intersect_levels;
    plan_->GetOrderedConnectivity(intersect_levels);
    AllCondType conditions;
    plan_->GetOrderedOrdering(conditions);

    std::vector<uintV> levels_new_nghrs(plan_->GetVertexCount());
    std::vector<uintV> levels_old_nghrs(plan_->GetVertexCount());
    auto &con = plan_->GetConnectivity();
    for (uintV level = 0; level < plan_->GetVertexCount(); ++level) {
      levels_old_nghrs[level] = 0;
      levels_new_nghrs[level] = 0;
      for (size_t pred_id = 0; pred_id < con[level].size(); ++pred_id) {
        auto l = con[level][pred_id];
        if (l > level) {
          bool in_neighborhood = false;
          for (size_t j = 0; j < con[l].size(); ++j) {
            if (con[l][j] < level) {
              in_neighborhood = true;
            }
          }
          if (in_neighborhood)
            levels_old_nghrs[level]++;
          else
            levels_new_nghrs[level]++;
        }
      }
    }
    uintV **paths = new uintV *[thread_num_];
    std::unordered_map<uintV, uintV> *in_neighborhood =
        new std::unordered_map<uintV, uintV>[thread_num_];
    for (size_t i = 0; i < thread_num_; ++i) {
      paths[i] = new uintV[plan_->GetVertexCount()];
    }
    omp_set_num_threads(thread_num_);

    TimeMeasurer timer;
    timer.StartTimer();

    long long total_match_count = 0;

#pragma omp parallel for schedule(dynamic) reduction(+ : total_match_count)
    for (uintV u = 0; u < graph_->GetVertexCount(); ++u) {
      long long ans = 0;
      size_t thread_id = omp_get_thread_num();
#if defined(PROFILE)
      TimeMeasurer timer2;
      timer2.StartTimer();
#endif
      in_neighborhood[thread_id].clear();
      paths[thread_id][0] = u;
      UpdateNeighborHood(u, in_neighborhood[thread_id]);
      DFS(thread_id, 1, paths[thread_id], ans, intersect_levels, conditions,
          levels_old_nghrs, levels_new_nghrs, in_neighborhood[thread_id]);
      Backtrack(u, in_neighborhood[thread_id]);
#if defined(PROFILE)
      timer2.EndTimer();
      thread_time_[thread_id] += timer2.GetElapsedMicroSeconds();
#endif
      total_match_count += ans;
    }

    timer.EndTimer();

    this->SetTotalMatchCount(total_match_count);
    std::cout << "total_match_count=" << total_match_count
              << ", elapsed_time=" << timer.GetElapsedMicroSeconds() / 1000.0
              << "ms" << std::endl;
    for (size_t i = 0; i < thread_num_; ++i) {
      delete[] paths[i];
      paths[i] = NULL;
    }
    delete[] paths;
    paths = NULL;
    delete[] in_neighborhood;
    in_neighborhood = NULL;
#if defined(PROFILE)
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      std::cout << "thread_id=" << thread_id
                << ",time=" << thread_time_[thread_id] / 1000.0 << "ms"
                << std::endl;
    }
#endif
  }

  private:
  void DFS(size_t thread_id, uintV cur_level, uintV *path,
           long long &ans,
           std::vector<std::vector<uintV>> &intersect_levels,
           AllCondType &conditions, std::vector<uintV> &levels_old_nghrs,
           std::vector<uintV> &levels_new_nghrs,
           std::unordered_map<uintV, uintV> &in_neighborhood) {
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
        UpdateNeighborHood(i, in_neighborhood);
        DFS(thread_id, cur_level + 1, path, ans, intersect_levels, conditions,
            levels_old_nghrs, levels_new_nghrs, in_neighborhood);
        Backtrack(i, in_neighborhood);
      }
    } else {
      // 3 rules described in "An in-depth comparison of
      // subgrah isomorphism algorithms in graph databases"
      // rule 1: connectivity
      auto row_ptrs = graph_->GetRowPtrs();
      auto cols = graph_->GetCols();
      std::vector<uintV> res[2];
      for (size_t j = 0; j < intersect_levels[cur_level].size(); ++j) {
        auto p2 = intersect_levels[cur_level][j];
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

      std::vector<uintV> &candidates =
          res[(intersect_levels[cur_level].size() + 1) % 2];
      for (size_t i = 0; i < candidates.size(); ++i) {
        if (CheckCondition(path, candidates[i], conditions[cur_level]) ==
                false ||
            CheckEquality(path, cur_level, candidates[i]))
          continue;
        size_t covered_count = 0, uncovered_count = 0;
        GetCoveredNeighborCount(candidates[i], path, cur_level, in_neighborhood,
                                covered_count, uncovered_count);
        // rule 2: #data vertices in the neighborhood path >=
        // #query vertices in the neighborhoold of matched levels
        if (covered_count < levels_old_nghrs[cur_level]) continue;
        // rule 3: #the neighbors of candidates[i] that are not matched
        // or neighbors of matched data vertices >=
        // #the neighbors of cur_level that are not matched or neighbors of
        // matched query vertices
        if (covered_count + uncovered_count - levels_old_nghrs[cur_level] <
            levels_new_nghrs[cur_level])
          continue;
        path[cur_level] = candidates[i];
        UpdateNeighborHood(candidates[i], in_neighborhood);
        DFS(thread_id, cur_level + 1, path, ans, intersect_levels, conditions,
            levels_old_nghrs, levels_new_nghrs, in_neighborhood);
        Backtrack(candidates[i], in_neighborhood);
      }
    }
  }

  void GetCoveredNeighborCount(uintV v, uintV *path, size_t path_length,
        std::unordered_map<uintV, uintV> &in_neighborhood,
                               size_t &covered_count,
                               size_t &uncovered_count) {
    covered_count = 0;
    uncovered_count = 0;
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();
    for (auto j = row_ptrs[v]; j < row_ptrs[v + 1]; ++j) {
      auto nv = cols[j];
      bool inpath = false;
      for (size_t l = 0; l < path_length; ++l) {
        if (path[l] == nv) {
          inpath = true;
          break;
        }
      }
      if (!inpath) {
        if (in_neighborhood.count(nv) && in_neighborhood[nv] > 0)
          covered_count++;
        else
          uncovered_count++;
      }
    }
  }
  void UpdateNeighborHood(
      uintV v, std::unordered_map<uintV, uintV> &in_neighborhood) {
    // in_neighborhood maintains the set of vertices connected with
    // the data vertices in the path
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();
    for (uintV j = row_ptrs[v]; j < row_ptrs[v + 1]; ++j) {
      auto nv = cols[j];
      if (in_neighborhood.count(nv) == 0) in_neighborhood[nv] = 0;
      in_neighborhood[nv] += 1;
    }
  }
  void Backtrack(uintV v,
                 std::unordered_map<uintV, uintV> &in_neighborhood) {
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();
    for (uintV j = row_ptrs[v]; j < row_ptrs[v + 1]; ++j) {
      auto nv = cols[j];
      in_neighborhood[nv] -= 1;
    }
  }

  private:
  TraversalPlan *plan_;
  Graph *graph_;
};

#endif
