#ifndef __HYBRID_LIGHT_PARTIAL_GROUP_CPU_COMPONENT_H__
#define __HYBRID_LIGHT_PARTIAL_GROUP_CPU_COMPONENT_H__

#include "HybridPtGroupCPUComp.h"
#include "LIGHTCPUCommon.h"
#include "LazyTraversalPlan.h"

using namespace LightCPU;

class HybridLIGHTCPUComponent : public HybridCPUComponent {
 public:
  HybridLIGHTCPUComponent(LazyTraversalPlan *plan,
                          TrackPartitionedGraph *cpu_rel,
                          bool materialize_result, size_t thread_num)
      : HybridCPUComponent(plan, cpu_rel, materialize_result, thread_num),
        plan_(plan) {
    work_meta_ = new WorkMeta();
#if defined(LIGHT_CPU_PROFILE)
    phase_profiler_->AddPhase("compute_time");
    phase_profiler_->AddPhase("count_time");
    phase_profiler_->AddPhase("filter_compute_time");
    phase_profiler_->AddPhase("process_time");
#endif
  }
  ~HybridLIGHTCPUComponent() {
    delete work_meta_;
    work_meta_ = NULL;
  }

  virtual void ThreadExecute(size_t thread_id, long long &ans, uintV v1,
                             uintV v2) {
    auto &group_dfs_ids = plan_->GetGroupDfsIds();
    for (size_t group_id = 0; group_id < group_dfs_ids.size(); ++group_id) {
      size_t dfs_id = group_dfs_ids[group_id][0];
      auto &exec_seq = plan_->GetInterPartitionExecuteOperations()[dfs_id];

      LightCPUWorkContext ctx;
      ctx.Init(group_id, dfs_id, &ans, plan_->GetVertexCount());
      ctx.path[exec_seq[0].second] = v1;
      ctx.path[exec_seq[2].second] = v2;

#if defined(DEBUG)
      assert(exec_seq[0].first == MATERIALIZE);
      assert(exec_seq[2].first == MATERIALIZE);
      assert(exec_seq[1].first == COMPUTE);
      assert(exec_seq[1].second == exec_seq[2].second);
#endif

#if defined(LIGHT_CPU_PROFILE)
      phase_profiler_->StartTimer("process_time", thread_id);
#endif
      Process(thread_id, work_meta_, &ctx, 2);
#if defined(LIGHT_CPU_PROFILE)
      phase_profiler_->EndTimer("process_time", thread_id);
#endif
    }
  }

 protected:
  void Process(size_t thread_id, WorkMeta *meta, LightCPUWorkContext *ctx,
               size_t cur_exec_level) {
    size_t group_id = ctx->group_id;
    size_t dfs_id = ctx->dfs_id;
    auto &exec_seq = plan_->GetInterPartitionExecuteOperations()[dfs_id];

    assert(cur_exec_level < exec_seq.size());
    auto op = exec_seq[cur_exec_level].first;
    uintV cur_u = exec_seq[cur_exec_level].second;

    uintV *path = ctx->path.data();
    auto &mvs =
        meta->group_meta[group_id].materialized_vertices[cur_exec_level];
    auto &umvs = meta->group_meta[group_id]
                     .computed_unmaterialized_vertices[cur_exec_level];
    auto &mcond = meta->group_meta[group_id].materialized_cond[cur_exec_level];

    switch (op) {
      case COMPUTE:
      case COMPUTE_COUNT: {
#if defined(LIGHT_CPU_PROFILE)
        phase_profiler_->StartTimer("compute_time", thread_id);
#endif
        auto &conn = meta->group_meta[group_id].conn[cur_u];
        auto &cand = ctx->candidates[cur_u];

        std::vector<uintV> intersect_result;
        MWayIntersect<CPUIntersectMethod::HOME_MADE>(
            path, cpu_relation_->GetRowPtrs(), cpu_relation_->GetCols(), conn,
            intersect_result);

        cand.clear();
        for (auto v : intersect_result) {
          if (!CheckDuplicate(path, v, conn,
                              cpu_relation_->GetVertexPartitionMap(),
                              exec_seq[0].second, exec_seq[2].second))
            continue;

          if (CheckCondition(path, v, mcond)) {
            // This condition contains NON_EQUAL check
            cand.push_back(v);
          }
        }
#if defined(LIGHT_CPU_PROFILE)
        phase_profiler_->EndTimer("compute_time", thread_id);
#endif

        if (op == COMPUTE) {
          Process(thread_id, meta, ctx, cur_exec_level + 1);
        } else {
#if defined(LIGHT_CPU_PROFILE)
          phase_profiler_->StartTimer("count_time", thread_id);
#endif
          Count(meta, ctx, cur_exec_level);
#if defined(LIGHT_CPU_PROFILE)
          phase_profiler_->EndTimer("count_time", thread_id);
#endif
        }
        cand.clear();

      } break;
      case MATERIALIZE: {
        if (cur_exec_level == 2) {
          if (CheckCondition(path, path[cur_u], mcond)) {
            Process(thread_id, meta, ctx, cur_exec_level + 1);
          }
        } else {
          for (auto v : ctx->candidates[cur_u]) {
            if (CheckCondition(path, v, mcond)) {
              path[cur_u] = v;
              Process(thread_id, meta, ctx, cur_exec_level + 1);
            }
          }
        }
      } break;
      case FILTER_COMPUTE: {
#if defined(LIGHT_CPU_PROFILE)
        phase_profiler_->StartTimer("filter_compute_time", thread_id);
#endif
        auto &cand = ctx->candidates[cur_u];
        std::vector<uintV> filter_result;
        for (auto v : cand) {
          if (CheckCondition(path, v, mcond)) {
            filter_result.push_back(v);
          }
        }
#if defined(LIGHT_CPU_PROFILE)
        phase_profiler_->EndTimer("filter_compute_time", thread_id);
#endif

        cand.swap(filter_result);
        Process(thread_id, meta, ctx, cur_exec_level + 1);
        cand.swap(filter_result);
      } break;
      case COUNT: {
#if defined(LIGHT_CPU_PROFILE)
        phase_profiler_->StartTimer("count_time", thread_id);
#endif
        Count(meta, ctx, cur_exec_level);
#if defined(LIGHT_CPU_PROFILE)
        phase_profiler_->EndTimer("count_time", thread_id);
#endif
      } break;
      default:
        assert(false);
        break;
    }
  }

  void Count(WorkMeta *meta, LightCPUWorkContext *ctx, size_t cur_exec_level) {
    long long *ans = ctx->ans;
    uintV *path = ctx->path.data();
    size_t group_id = ctx->group_id;
    for (auto dfs_id : plan_->GetGroupDfsIds()[group_id]) {
      auto &mvs =
          meta->group_meta[group_id].materialized_vertices[cur_exec_level];
      auto &umvs = meta->group_meta[group_id]
                       .computed_unmaterialized_vertices[cur_exec_level];
      auto &ccond = meta->dfs_meta[dfs_id].count_cond;

      auto &exec_seq = plan_->GetInterPartitionExecuteOperations()[dfs_id];
      auto op = exec_seq[cur_exec_level].first;
      auto cur_u = exec_seq[cur_exec_level].second;

      // check condition among materialized vertices
      bool path_valid = true;
      for (auto u : mvs) {
        if (!CheckCondition(path, path[u], ccond[u])) {
          path_valid = false;
          break;
        }
      }
      if (!path_valid) continue;

      // check condition between unmaterialized and materialized vertices
      LayeredArray<uintV> cands(ctx->candidates.size());
      std::vector<uintV> cur_umvs(umvs);
      if (op == COMPUTE_COUNT) cur_umvs.push_back(cur_u);
      for (auto u : cur_umvs) {
        for (auto v : ctx->candidates[u]) {
          if (CheckCondition(path, v, ccond[u])) {
            cands[u].push_back(v);
          }
        }
      }

      if (cur_umvs.size() == 1) {
        *ans += cands[cur_umvs[0]].size();
      } else if (cur_umvs.size() == 2) {
        // check condition among unmaterialized vertices
        auto cond_op = Plan::GetConditionType(cur_umvs[0], cur_umvs[1],
                                              meta->dfs_meta[dfs_id].cond);
        *ans += ComputeCount(
            cands[cur_umvs[0]].data(), cands[cur_umvs[0]].size(),
            cands[cur_umvs[1]].data(), cands[cur_umvs[1]].size(), cond_op);

      } else {
        assert(false);
      }
    }
  }

  static size_t ComputeCount(uintV *cand1, size_t cand1_count, uintV *cand2,
                             size_t cand2_count, CondOperator op) {
    size_t ret = 0;
    for (size_t i = 0; i < cand1_count; ++i) {
      uintV e = cand1[i];
      size_t pos = std::lower_bound(cand2, cand2 + cand2_count, e) - cand2;
      bool equal = (pos < cand2_count && cand2[pos] == e);
      switch (op) {
        case LESS_THAN:
          pos += equal ? 1 : 0;
          ret += cand2_count - pos;
          break;
        case LARGER_THAN:
          ret += pos;
          break;
        case NON_EQUAL:
          ret += cand2_count - (equal ? 1 : 0);
          break;
        default:
          break;
      }
    }
    return ret;
  }

  virtual void Init() {
    work_meta_->Init(plan_);
#if defined(DEBUG)
    work_meta_->Print();
#endif
  }

 private:
  LazyTraversalPlan *plan_;
  WorkMeta *work_meta_;
};

#endif