#ifndef __QUERY_OPTIMIZER_LAZY_TRAVERSAL_PLAN_GENERATOR_H__
#define __QUERY_OPTIMIZER_LAZY_TRAVERSAL_PLAN_GENERATOR_H__

#include "LazyTraversalCommon.h"
#include "LazyTraversalPlanUtils.h"

class LazyTraversalPlanGenerator {
 public:
  // For the last two vertices, apply the optimization:
  // avoid materialization as much as possible.
  static void GenerateCount2ExecuteSequence(
      std::vector<LazyTraversalEntry>& exec_seq, const SearchSequence& seq,
      const std::vector<uintV>& unmaterialized_vertices,
      const std::vector<bool>& need_filter_compute,
      LazyTraversalCompressLevel opt_level) {
    size_t vertex_count = seq.size();
    if (opt_level == COMPRESS_LEVEL_MATERIALIZE) {
      exec_seq.push_back(
          std::make_pair(MATERIALIZE, unmaterialized_vertices[0]));
      exec_seq.push_back(
          std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[1]));
      exec_seq.push_back(std::make_pair(COUNT, vertex_count));
    } else if (opt_level == COMPRESS_LEVEL_NON_MATERIALIZE) {
      if (need_filter_compute[0]) {
        exec_seq.push_back(
            std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[0]));
      }
      if (need_filter_compute[1]) {
        exec_seq.push_back(
            std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[1]));
      }
      exec_seq.push_back(std::make_pair(COUNT, vertex_count));
      return;
    } else if (opt_level == COMPRESS_LEVEL_NON_MATERIALIZE_OPT) {
      if (need_filter_compute[0] && need_filter_compute[1]) {
        // TODO: FILTER_COMPUTE_COUNT
        // the better way is to just materialize the candidate set of one
        // vertex, and do not materialize the other one but simply count it
        exec_seq.push_back(
            std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[0]));
        exec_seq.push_back(
            std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[1]));
        exec_seq.push_back(std::make_pair(COUNT, vertex_count));
      } else if (!need_filter_compute[0] && !need_filter_compute[1]) {
        // for unmaterialized_vertices, the last operations could be
        // COMPUTE or FILTER_COMPUTE
        size_t last_index0 = exec_seq.size();
        size_t last_index1 = exec_seq.size();
        for (int j = exec_seq.size() - 1; j >= 0; --j) {
          if (exec_seq[j].second == unmaterialized_vertices[0]) {
            last_index0 = j;
          }
          if (exec_seq[j].second == unmaterialized_vertices[1]) {
            last_index1 = j;
          }
        }
        size_t max_index = std::max(last_index0, last_index1);
        assert(max_index == exec_seq.size() - 1);
        exec_seq[max_index].first = COMPUTE_COUNT;
      } else {
        // one need_filter_compute and another do not
        //
        uintV need_filter_u;
        uintV non_need_filter_u;
        if (need_filter_compute[0]) {
          need_filter_u = unmaterialized_vertices[0];
          non_need_filter_u = unmaterialized_vertices[1];
        } else {
          need_filter_u = unmaterialized_vertices[1];
          non_need_filter_u = unmaterialized_vertices[0];
        }

        size_t non_need_filter_u_index =
            LazyTraversalPlanUtils::GetOperationIndex(exec_seq, COMPUTE,
                                                      non_need_filter_u);
        exec_seq.erase(exec_seq.begin() + non_need_filter_u_index);

        exec_seq.push_back(std::make_pair(FILTER_COMPUTE, need_filter_u));
        exec_seq.push_back(std::make_pair(COMPUTE_COUNT, non_need_filter_u));
      }
    } else {
      assert(false);
    }
  }

  // For the last vertex, fast counting.
  static void GenerateCount1ExecuteSequence(
      std::vector<LazyTraversalEntry>& exec_seq, const SearchSequence& seq,
      const std::vector<uintV>& unmaterialized_vertices,
      const std::vector<bool>& need_filter_compute,
      LazyTraversalCompressLevel opt_level) {
    size_t vertex_count = seq.size();
    if (opt_level == COMPRESS_LEVEL_MATERIALIZE ||
        opt_level == COMPRESS_LEVEL_NON_MATERIALIZE) {
      if (need_filter_compute[0]) {
        exec_seq.push_back(
            std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[0]));
      }
      exec_seq.push_back(std::make_pair(COUNT, vertex_count));

    } else if (opt_level == COMPRESS_LEVEL_NON_MATERIALIZE_OPT) {
      if (need_filter_compute[0]) {
        // TODO: a better way is not to materialize but only count and reduce.
        // FILTER_COMPUTE_COUNT
        exec_seq.push_back(
            std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[0]));
        exec_seq.push_back(std::make_pair(COUNT, vertex_count));
      } else {
        // use COMPUTE_COUNT directly to avoid materializing the
        // candidate set of the last pattern vertex
        auto u = unmaterialized_vertices[0];
        size_t last_index = exec_seq.size() - 1;
        assert(exec_seq[last_index].second == u &&
               exec_seq[last_index].first == COMPUTE);
        exec_seq[last_index].first = COMPUTE_COUNT;
      }
    } else {
      assert(false);
    }
  }

  // Generate the sequences that compute all vertices.
  // This is the routine used to generate the execute sequence.
  static void GenerateComputeExecuteSequence(
      std::vector<LazyTraversalEntry>& exec_seq, const SearchSequence& seq,
      const AllConnType& conn, size_t vertex_count,
      bool materialize_first_edge) {
    AllConnType backward_conn;
    Plan::GetBackwardConnectivity(backward_conn, conn, seq);

    std::vector<bool> materialized(vertex_count, false);
    auto first_vertex = seq[0];
    exec_seq.push_back(std::make_pair(MATERIALIZE, first_vertex));
    materialized[first_vertex] = true;

    // compute each pattern vertex, materialize the backward neighbors if
    // necessary
    for (uintV i = 1; i < seq.size(); ++i) {
      auto u = seq[i];
      if (materialize_first_edge && i == 1) {
        exec_seq.push_back(std::make_pair(COMPUTE, u));
        exec_seq.push_back(std::make_pair(MATERIALIZE, u));
        materialized[u] = true;
      } else {
        for (auto prev_u : backward_conn[u]) {
          if (!materialized[prev_u]) {
            exec_seq.push_back(std::make_pair(MATERIALIZE, prev_u));
            materialized[prev_u] = true;
          }
        }
        exec_seq.push_back(std::make_pair(COMPUTE, u));
      }
    }
  }

  // Discover the status of unmaterialized vertices.
  // This is the routine used before generate the count execute sequence.
  static void InspectUnmaterializedVertices(
      VTGroup& unmaterialized_vertices, std::vector<bool>& need_filter_compute,
      const std::vector<LazyTraversalEntry>& exec_seq, size_t vertex_count) {
    std::vector<bool> materialized(vertex_count, false);
    for (auto entry : exec_seq) {
      if (entry.first == MATERIALIZE) {
        materialized[entry.second] = true;
      }
    }

    for (uintV u = 0; u < vertex_count; ++u) {
      if (!materialized[u]) {
        unmaterialized_vertices.push_back(u);

        size_t ind =
            LazyTraversalPlanUtils::GetOperationIndex(exec_seq, COMPUTE, u);
        // in the range (ind, end), if there is any materialized vertices,
        bool f = false;
        for (size_t i = ind + 1; i < exec_seq.size(); ++i) {
          if (exec_seq[i].first == MATERIALIZE) {
            f = true;
          }
        }
        need_filter_compute.push_back(f);
      }
    }
  }

  // General method of execute sequence generation.
  // extreme_opt is set to false for inter-partition search to ease the
  // counting, so that the execute sequence does not contain COMPUTE_PATH_COUNT,
  // COMPUTE_COUNT.
  // materialize_first_edge: usually set to true for inter-partition search.
  // In this case, the first edge is ensured to be materialized and we can
  // check duplicates.
  static void GenerateExecuteSequence(std::vector<LazyTraversalEntry>& exec_seq,
                                      const SearchSequence& seq,
                                      const AllConnType& conn,
                                      size_t vertex_count,
                                      LazyTraversalCompressLevel opt_level,
                                      bool materialize_first_edge) {
    GenerateComputeExecuteSequence(exec_seq, seq, conn, vertex_count,
                                   materialize_first_edge);

    // now we have computed all pattern vertices and materialized some vertices
    // optimize the counting process

    // the vertices that have been computed but not materialized
    VTGroup unmaterialized_vertices;
    // whether a unmaterialized vertex need filter compute:
    // needed when there are new vertices materialized after
    // this vertex is computed.
    // This means that the computed candidate set of this vertex violate
    // may violate the non-equality constraint and the constraints
    // should be re-checked again.
    std::vector<bool> need_filter_compute;

    InspectUnmaterializedVertices(unmaterialized_vertices, need_filter_compute,
                                  exec_seq, vertex_count);

    if (unmaterialized_vertices.size() == 1) {
      GenerateCount1ExecuteSequence(exec_seq, seq, unmaterialized_vertices,
                                    need_filter_compute, opt_level);
    } else if (unmaterialized_vertices.size() == 2) {
      GenerateCount2ExecuteSequence(exec_seq, seq, unmaterialized_vertices,
                                    need_filter_compute, opt_level);
    } else {
      // the most general case
      // more than two unmaterialized vertices, have to materialize the
      // pattern vertices one by one and then count.

      for (size_t i = 2; i < unmaterialized_vertices.size(); ++i) {
        auto u = unmaterialized_vertices[i];
        exec_seq.push_back(std::make_pair(MATERIALIZE, u));
      }

      std::vector<uintV> remaining_unmaterialized_vertices(
          unmaterialized_vertices.begin(), unmaterialized_vertices.begin() + 2);
      std::vector<bool> remaining_need_filter_compute(2, true);
      GenerateCount2ExecuteSequence(exec_seq, seq,
                                    remaining_unmaterialized_vertices,
                                    remaining_need_filter_compute, opt_level);
    }
  }

  // Specialized fast counting for some queries.
  static void SpecialExecuteSequence(std::vector<LazyTraversalEntry>& exec_seq,
                                     const QueryType query_type,
                                     const SearchSequence& seq,
                                     const AllConnType& conn,
                                     size_t vertex_count,
                                     bool materialize_first_edge) {
    GenerateComputeExecuteSequence(exec_seq, seq, conn, vertex_count,
                                   materialize_first_edge);

    // the vertices that have been computed but not materialized
    VTGroup unmaterialized_vertices;
    // whether a unmaterialized vertex need filter compute:
    // needed when there are new vertices materialized after
    // this vertex is computed.
    // This means that the computed candidate set of this vertex violate
    // may violate the non-equality constraint and the constraints
    // should be re-checked again.
    std::vector<bool> need_filter_compute;

    InspectUnmaterializedVertices(unmaterialized_vertices, need_filter_compute,
                                  exec_seq, vertex_count);

    if (query_type == Q2) {
      assert(unmaterialized_vertices.size() == 2);
      size_t last2 = exec_seq.size() - 2;
      size_t last1 = exec_seq.size() - 1;
      assert(exec_seq[last1].first == COMPUTE &&
             exec_seq[last2].first == COMPUTE);
      exec_seq[last2].first = COMPUTE_PATH_COUNT;
      exec_seq[last1].first = COMPUTE_PATH_COUNT;
      exec_seq.push_back(std::make_pair(COUNT, vertex_count));

    } else if (query_type == Q6) {
      // for Q6, can simply keep the number for each path instead of
      // materializing the candidate set for each path

      assert(unmaterialized_vertices.size() == 2);
      size_t last2 = exec_seq.size() - 2;
      size_t last1 = exec_seq.size() - 1;
      assert(exec_seq[last1].first == COMPUTE &&
             exec_seq[last2].first == COMPUTE);
      exec_seq[last2].first = COMPUTE_PATH_COUNT;
      exec_seq[last1].first = COMPUTE_PATH_COUNT;
      exec_seq.push_back(std::make_pair(COUNT, vertex_count));

    } else if (query_type == Q5) {
      for (size_t i = 0; i < unmaterialized_vertices.size(); ++i) {
        if (need_filter_compute[i]) {
          exec_seq.push_back(
              std::make_pair(FILTER_COMPUTE, unmaterialized_vertices[i]));
        }
      }
      exec_seq.push_back(std::make_pair(COUNT, vertex_count));
    } else {
      assert(false);
    }
  }
};

#endif
