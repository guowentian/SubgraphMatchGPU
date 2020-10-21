#ifndef __QUERY_OPTIMIZER_LAZY_TRAVERSAL_PLAN_UTILS_H__
#define __QUERY_OPTIMIZER_LAZY_TRAVERSAL_PLAN_UTILS_H__

#include "LazyTraversalCommon.h"
#include "Plan.h"

class LazyTraversalPlanUtils {
 public:
  static size_t GetOperationIndex(
      const std::vector<LazyTraversalEntry>& exec_seq,
      LazyTraversalOperation op, uintV u) {
    for (size_t i = 0; i < exec_seq.size(); ++i) {
      if (exec_seq[i].first == op && exec_seq[i].second == u) {
        return i;
      }
    }
    return exec_seq.size();
  }

  // Transform the original vertex ids into the indices for the execute
  // sequence. Based on the search sequence
  static void GetIndexBasedExecuteSequence(
      std::vector<LazyTraversalEntry>& exec_seq, const SearchSequence& seq,
      size_t vertex_count) {
    std::vector<uintV> otn_map(vertex_count, vertex_count);
    for (size_t i = 0; i < seq.size(); ++i) {
      auto u = seq[i];
      otn_map[u] = i;
    }

    for (size_t i = 0; i < exec_seq.size(); ++i) {
      if (exec_seq[i].second < vertex_count) {
        exec_seq[i].second = otn_map[exec_seq[i].second];
      } else {
        assert(exec_seq[i].second == vertex_count);
      }
    }
  }

  static void GenerateConstraints(
      MultiVTGroup& materialized_vertices,
      MultiVTGroup& computed_unmaterialized_vertices,
      const std::vector<LazyTraversalEntry>& exec_seq, size_t vertex_count) {
    std::vector<bool> materialized(vertex_count, false);
    materialized.resize(vertex_count, false);
    VTGroup all_materialized_vertices;
    VTGroup all_computed_vertices;

    materialized_vertices.resize(exec_seq.size());
    computed_unmaterialized_vertices.resize(exec_seq.size());
    for (size_t i = 0; i < exec_seq.size(); ++i) {
      materialized_vertices[i].assign(all_materialized_vertices.begin(),
                                      all_materialized_vertices.end());
      for (auto u : all_computed_vertices) {
        if (!materialized[u]) {
          computed_unmaterialized_vertices[i].push_back(u);
        }
      }

      if (exec_seq[i].first == COMPUTE ||
          exec_seq[i].first == COMPUTE_PATH_COUNT) {
        all_computed_vertices.push_back(exec_seq[i].second);
      } else if (exec_seq[i].first == MATERIALIZE) {
        all_materialized_vertices.push_back(exec_seq[i].second);
        materialized[exec_seq[i].second] = true;
      }
    }
  }

  // the ordering constraints enforced for COMPUTE step,
  // i.e., the ordering for those vertices that have been materialized
  // before this vertex is computed.
  static void GetComputeCondition(
      AllCondType& ret, const std::vector<LazyTraversalEntry>& exec_seq,
      const AllCondType& order, size_t vertex_count) {
    VTGroup materialized_vertices;
    ret.resize(vertex_count);
    for (size_t i = 0; i < exec_seq.size(); ++i) {
      if (exec_seq[i].first == COMPUTE || exec_seq[i].first == COMPUTE_COUNT ||
          exec_seq[i].first == COMPUTE_PATH_COUNT) {
        auto u = exec_seq[i].second;
        ret[u].clear();

        for (auto prev_u : materialized_vertices) {
          CondOperator op = Plan::GetConditionType(u, prev_u, order);
          ret[u].push_back(std::make_pair(op, prev_u));
        }
      } else if (exec_seq[i].first == MATERIALIZE) {
        materialized_vertices.push_back(exec_seq[i].second);
      }
    }
  }

  // The ordering constraints enforced for MATERIALIZE step.
  // The ordering for those vertices that are materialized after
  // this vertex is computed and before this vertex is materialized.
  static void GetMaterializeCondition(
      AllCondType& ret, const std::vector<LazyTraversalEntry>& exec_seq,
      const AllCondType& order, size_t vertex_count) {
    ret.resize(vertex_count);
    for (size_t i = 0; i < exec_seq.size(); ++i) {
      if (exec_seq[i].first == MATERIALIZE) {
        auto u = exec_seq[i].second;
        ret[u].clear();

        // after the time u is computed and before the time u is materialized,
        // the set of vertices that are materialized
        VTGroup independence_vertices;
        for (int j = (int)i - 1; j >= 0; --j) {
          if (exec_seq[j].first == COMPUTE && exec_seq[j].second == u) {
            break;
          }
          if (exec_seq[j].first == MATERIALIZE) {
            auto prev_u = exec_seq[j].second;
            independence_vertices.push_back(prev_u);
          }
        }

        for (auto prev_u : independence_vertices) {
          CondOperator op = Plan::GetConditionType(u, prev_u, order);
          ret[u].push_back(std::make_pair(op, prev_u));
        }
      }
    }
  }

  // The ordering constraints used for FILTER_COMPUTE step.
  // The ordering for those vertices that are materialized after
  // the last time when this vertex is computed
  // and before this vertex is filter-computed
  static void GetFilterCondition(
      AllCondType& ret, const std::vector<LazyTraversalEntry>& exec_seq,
      const AllCondType& order, size_t vertex_count) {
    ret.resize(vertex_count);
    for (size_t i = 0; i < exec_seq.size(); ++i) {
      if (exec_seq[i].first == FILTER_COMPUTE) {
        auto u = exec_seq[i].second;
        ret[u].clear();

        // after the time u is computed and before the time u is filter,
        // the set of vertices that are materialized.
        VTGroup M;
        for (int j = (int)i - 1; j >= 0; --j) {
          if (exec_seq[j].first == COMPUTE && exec_seq[j].second == u) {
            break;
          }
          if (exec_seq[j].first == MATERIALIZE) {
            M.push_back(exec_seq[j].second);
          }
        }

        for (auto prev_u : M) {
          CondOperator op = Plan::GetConditionType(u, prev_u, order);
          ret[u].push_back(std::make_pair(op, prev_u));
        }
      }
    }
  }

  // For all vertices, Get the ordering to all materialized vertices .
  // This is needed when counting (inter-partition) instances.
  // At this time, we need to apply ordering constraints to those materialized
  // vertices.
  static void GetCountToMaterializedVerticesCondition(
      AllCondType& ret, const std::vector<LazyTraversalEntry>& exec_seq,
      const AllCondType& order, size_t vertex_count) {
    ret.resize(vertex_count);
    VTGroup materialized_vertices;
    for (auto p : exec_seq) {
      if (p.first == MATERIALIZE) {
        materialized_vertices.push_back(p.second);
      }
    }

    for (uintV u = 0; u < vertex_count; ++u) {
      ret[u].clear();
      for (auto mu : materialized_vertices) {
        if (u != mu) {
          CondOperator op = Plan::GetConditionType(u, mu, order);
          ret[u].push_back(std::make_pair(op, mu));
        }
      }
    }
  }

  static void GetMaterializedSequenceAtCompute(
      MultiVTGroup& comp_mseq, std::vector<LazyTraversalEntry>& exec_seq,
      size_t vertex_count) {
    comp_mseq.resize(vertex_count);
    VTGroup materialized_vertices;
    for (auto entry : exec_seq) {
      if (entry.first == MATERIALIZE) {
        materialized_vertices.push_back(entry.second);
      } else if (entry.first == COMPUTE || entry.first == COMPUTE_PATH_COUNT ||
                 entry.first == COMPUTE_COUNT) {
        comp_mseq[entry.second].assign(materialized_vertices.begin(),
                                       materialized_vertices.end());
      }
    }
  }

  static void GetPrevMaterializeVertex(
      std::vector<uintV>& prev_mvertex,
      std::vector<LazyTraversalEntry>& exec_seq, size_t vertex_count) {
    prev_mvertex.resize(exec_seq.size());
    uintV last_mvertex = vertex_count;
    for (size_t i = 0; i < exec_seq.size(); ++i) {
      prev_mvertex[i] = last_mvertex;
      if (exec_seq[i].first == MATERIALIZE) {
        last_mvertex = exec_seq[i].second;
      }
    }
  }
};

#endif
