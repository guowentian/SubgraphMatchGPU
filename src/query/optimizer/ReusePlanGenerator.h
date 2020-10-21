#ifndef __QUERY_OPTIMIZER_REUSE_PLAN_GENERATOR_H__
#define __QUERY_OPTIMIZER_REUSE_PLAN_GENERATOR_H__

#include "Plan.h"
#include "ReuseTraversalCommon.h"
#include "TraversalOptimizer.h"

#include <algorithm>
#include <cmath>
#include <string>

struct ReuseVertexPlanElement {
  ConnType covered_;
  double cost_;
  // attachment
  // mapping
  VertexMapping g_;
  uintV uj_;
};

class MinimumSetCoverSolver {
 public:
  static void MinSetCover(VTGroup& universe,
                          std::vector<ReuseVertexPlanElement>& candidates,
                          std::vector<ReuseVertexPlanElement>& ans) {
    size_t m = candidates.size();
    assert(m < 32);

    size_t universe_mask = 0;
    for (auto u : universe) {
      universe_mask |= (1ULL << u);
    }

    // initialize ans as the plan of using each separate backward neighbor
    for (auto u : universe) {
      ReuseVertexPlanElement cand;
      cand.covered_.push_back(u);
      cand.cost_ = 0.0;
      ans.push_back(cand);
    }
    double cur_ans_cost = 0.0;

    for (size_t set_count = 1; set_count <= universe.size(); ++set_count) {
      if (set_count > ans.size()) {
        // no more better result
        break;
      }
      // choose set_count elements from m elements in candidates
      // enumerate all combinations
      /*#if defined(DEBUG)
            std::cout << "find set cover, set_count=" << set_count << std::endl;
      #endif
      */
      std::string bitmask(set_count, 1);
      bitmask.resize(m, 0);
      do {
        size_t covered_mask = 0;
        double cur_cost = 0.0;
        for (size_t i = 0; i < bitmask.size(); ++i) {
          if (bitmask[i]) {
            auto& cand = candidates[i];
            for (auto u : cand.covered_) {
              covered_mask |= (1ULL << u);
              cur_cost += cand.cost_;
            }
          }
        }

        // if can cover all backward neighbors
        if ((covered_mask & universe_mask) == universe_mask) {
          if (set_count < ans.size() ||
              (set_count == ans.size() && cur_cost < cur_ans_cost)) {
            ans.clear();
            for (size_t i = 0; i < bitmask.size(); ++i) {
              if (bitmask[i]) {
                auto& cand = candidates[i];
                ans.push_back(cand);
              }
            }
            cur_ans_cost = cur_cost;
          }
        }
      } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    }
  }

  static void MinSetCoverSlow(VTGroup& universe,
                              std::vector<ReuseVertexPlanElement>& candidates,
                              std::vector<ReuseVertexPlanElement>& ans) {
    // initialize ans as the plan of using each separate backward neighbor
    std::sort(universe.begin(), universe.end());
    for (auto u : universe) {
      ReuseVertexPlanElement cand;
      cand.covered_.push_back(u);
      cand.cost_ = 0.0;
      ans.push_back(cand);
    }
    double cur_ans_cost = 0.0;

    size_t m = candidates.size();
    assert(m < 32);
    for (size_t i = 1; i < (1ULL << m); ++i) {
      size_t cur_set_count = Popcount(i, m);
      if (cur_set_count <= ans.size()) {
        std::vector<uintV> all_covered;
        double cur_cost = 0.0;
        size_t state = i;
        for (size_t j = 0; j < m; ++j) {
          if ((state >> j) & 1) {
            auto& cand = candidates[j];
            all_covered.insert(all_covered.end(), cand.covered_.begin(),
                               cand.covered_.end());
            cur_cost += cand.cost_;
          }
        }

        std::sort(all_covered.begin(), all_covered.end());
        std::unique(all_covered.begin(), all_covered.end());

        if (Equal(universe, all_covered)) {
          // can cover the backward neighbors
          if (cur_set_count < ans.size() ||
              (cur_set_count == ans.size() && cur_cost < cur_ans_cost)) {
            ans.clear();
            // assign
            for (size_t j = 0; j < m; ++j) {
              if ((state >> j) & 1) {
                ans.push_back(candidates[j]);
              }
            }
            cur_ans_cost = cur_cost;
          }
        }
      }
    }
  }

  static size_t Popcount(size_t state, size_t n) {
    size_t ret = 0;
    for (size_t i = 0; i < n; ++i) {
      if (state & 1) {
        ret++;
      }
      state >>= 1;
    }
    return ret;
  }
  static bool Equal(VTGroup& pattern, VTGroup& data) {
    if (pattern.size() != data.size()) return false;
    for (size_t i = 0; i < data.size(); ++i) {
      if (pattern[i] != data[i]) {
        return false;
      }
    }
    return true;
  }
};

class ReusePlanGenerator {
 public:
  // Generate the reusable vertex plan for all vertices
  static void GenerateReuseTraversalPlan(
      const SearchSequence& seq, const AllConnType& con,
      double avg_size_tree_node,
      std::vector<VertexReuseIntersectPlan>& reuse_plan) {
    for (size_t i = 0; i < seq.size(); ++i) {
      VertexReuseIntersectPlan reuse_vertex_plan;
      GenerateReuseTraversalVertexPlan(seq, con, i, avg_size_tree_node,
                                       reuse_vertex_plan);
      reuse_plan.push_back(reuse_vertex_plan);
    }
  }

  // Generate reuse vertex plan for the vertex seq[i]
  static void GenerateReuseTraversalVertexPlan(
      const SearchSequence& seq, const AllConnType& con, size_t i,
      double avg_size_tree_node, VertexReuseIntersectPlan& reuse_vertex_plan) {
    AllConnType backward_conn;
    Plan::GetBackwardConnectivity(backward_conn, con, seq);

    std::vector<ReuseVertexPlanElement> candidates;
    // initialize as each single backward neighbor
    for (auto u : backward_conn[seq[i]]) {
      ReuseVertexPlanElement cand;
      cand.covered_.push_back(u);
      cand.cost_ = 0.0;
      candidates.push_back(cand);
    }

    for (size_t j = 0; j < i; ++j) {
      if (backward_conn[seq[j]].size() > 1) {
        // in this case, if ui constraint subsume uj, the mapping g
        // can cover more than 1 backward neighbor of ui
        std::vector<VertexMapping> mappings;
        Subsume(seq, con, seq[i], seq[j], mappings);
        for (size_t g_index = 0; g_index < mappings.size(); ++g_index) {
          auto& g = mappings[g_index];
          // for each mapping 'g' such that seq[i] constraint subsume
          // seq[j] with g

          ReuseVertexPlanElement cand;
          for (auto u : backward_conn[seq[j]]) {
            cand.covered_.push_back(g[u]);
          }
          cand.g_.assign(g.begin(), g.end());
          cand.uj_ = seq[j];
          cand.cost_ =
              TreeSearchCost(seq, con, seq[i], seq[j], g, avg_size_tree_node);
          candidates.push_back(cand);
        }
      }
    }

    // when the pattern graph is large, we need a faster (maybe approximate)
    // minimum set cover solver
    std::vector<ReuseVertexPlanElement> ans;
    MinimumSetCoverSolver::MinSetCover(backward_conn[seq[i]], candidates, ans);

    // assign the answer
    for (auto& cand : ans) {
      if (cand.covered_.size() == 1) {
        reuse_vertex_plan.separate_conn_.push_back(cand.covered_[0]);
      } else {
        ConnType reuse_conn(cand.covered_.begin(), cand.covered_.end());
        auto source_vertex = cand.uj_;
        ConnType source_conn(backward_conn[cand.uj_].begin(),
                             backward_conn[cand.uj_].end());
        VertexMapping& mapping = cand.g_;
        VertexMapping inverted_mapping(mapping.size(), mapping.size());
        for (uintV u = 0; u < mapping.size(); ++u) {
          if (mapping[u] < mapping.size()) {
            inverted_mapping[mapping[u]] = u;
          }
        }

        ReuseConnMeta reuse_conn_meta;
        reuse_conn_meta.Init(reuse_conn, source_vertex, source_conn, mapping,
                             inverted_mapping);
        reuse_vertex_plan.GetReuseConnectivityMeta().push_back(reuse_conn_meta);
      }
    }
  }

  // Estimate the cost of the searching over tree index.
  // Given ui constraint subsume uj with g.
  // avg_size_tree_node is the average size of the tree node,
  // which is estimated as average node degree,
  static double TreeSearchCost(const SearchSequence& seq,
                               const AllConnType& con, uintV ui, uintV uj,
                               const VertexMapping& g,
                               double avg_size_tree_node) {
    size_t n = seq.size();
    std::vector<uintV> rank(n, n);
    for (size_t i = 0; i < seq.size(); ++i) {
      rank[seq[i]] = i;
    }

    size_t w = 0;
    for (; w < rank[uj]; ++w) {
      auto u = seq[w];
      if (g[u] != u) {
        break;
      }
    }
    // for i in [0,w), there is g[seq[i]] = seq[i]

    if (w == 0) {
      // w non-exists
      size_t j = rank[uj];
      return j * log2(avg_size_tree_node);
    } else {
      // w exists
      double ret = rank[ui] - w;
      size_t j = rank[uj];
      if (j > w) {
        ret += (j - w) * log2(avg_size_tree_node);
      }
      return ret;
    }
  }

  // Given the match order 'seq', find whether u1 constraint subsume u2.
  // The all possible mappings (from u2 to u1) are stored in 'mappings'
  // The mapping is based on original vertex id, of size n.
  static void Subsume(const SearchSequence& seq, const AllConnType& con,
                      uintV u1, uintV u2,
                      std::vector<VertexMapping>& mappings) {
    // rank
    size_t n = seq.size();
    std::vector<uintV> rank(n, n);
    for (size_t i = 0; i < n; ++i) {
      rank[seq[i]] = i;
    }

    mappings.clear();
    if (rank[u1] < rank[u2]) {
      return;
    }

    // enumerate all possible mappings from V(P_j) -> V(P_i)
    // where j = rank(u2), i = rank(u1).
    // This is equivalent to all combinations of choosing j items
    // from i total items.

    AllConnType backward_conn;
    Plan::GetBackwardConnectivity(backward_conn, con, seq);

    // a sequence with (j+1 ) 1s and (i-j) 0s
    std::string bitmask(rank[u2] + 1, 1);
    bitmask.resize(rank[u1] + 1, 0);

    do {
      // the positions where 1 appears
      std::vector<uintV> enumerated_positions;
      for (size_t i = 0; i < bitmask.size(); ++i) {
        if (bitmask[i]) {
          enumerated_positions.push_back(i);
        }
      }
      assert(enumerated_positions.size() == rank[u2] + 1);

      // g is the current enumerated mapping
      std::vector<uintV> g(n, n);
      for (size_t j = 0; j < enumerated_positions.size(); ++j) {
        auto from = seq[j];
        auto to = seq[enumerated_positions[j]];
        g[from] = to;
      }

      /*#if defined(DEBUG)
            for (size_t j = 0; j < enumerated_positions.size(); ++j) {
              auto from = seq[j];
              auto to = seq[enumerated_positions[j]];
              if (j > 0) std::cout << ",";
              std::cout << "(" << from << "," << to << ")";
            }
            std::cout << std::endl;
      #endif
      */

      bool valid = true;
      // specifically map u2 to u1
      if (g[u2] != u1) valid = false;
      // backward neighbor containment
      for (size_t j = 0; j <= rank[u2]; ++j) {
        auto from = seq[j];
        auto to = g[from];
        for (auto prev_u : backward_conn[from]) {
          auto to_prev_u = g[prev_u];
          // to_prev_u should be the backward neighbor of to
          bool exists = false;
          for (auto u : backward_conn[to]) {
            if (u == to_prev_u) {
              exists = true;
              break;
            }
          }
          if (!exists) {
            valid = false;
          }
        }
      }
      // order preserving
      for (size_t j = 0; j < rank[u2]; ++j) {
        auto from0 = seq[j];
        auto from1 = seq[j + 1];
        if (rank[g[from0]] < rank[g[from1]])
          ;
        else {
          valid = false;
        }
      }

      if (valid) {
        mappings.push_back(g);
      }

    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
  }
};

#endif
