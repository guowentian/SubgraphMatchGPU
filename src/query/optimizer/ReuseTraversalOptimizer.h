#ifndef __QUERY_REUSE_TRAVERSAL_OPTIMIZER_H__
#define __QUERY_REUSE_TRAVERSAL_OPTIMIZER_H__

#include "ReusePlanGenerator.h"

class ReuseTraversalOptimizer : public TraversalOptimizer {
 public:
  ReuseTraversalOptimizer(QueryType query_type, size_t vertex_count,
                          AllConnType& con, AllCondType& order)
      : TraversalOptimizer(query_type, vertex_count, con, order) {}

  virtual void GenerateOrder() {
    // parameters
    const double avg_degree = kAvgVertexDegree;
    const double ratio_random_access_versus_seq_access =
        kRatioRandomAccessVersusSeqAccess;
    const double avg_size_tree_node = kAvgSizeTreeNode;

    // enumerate each possible order,
    // choose the one with lower cost
    size_t n = vertex_count_;
    SearchSequence seq;
    for (size_t i = 0; i < n; ++i) {
      seq.push_back(i);
    }

    SearchSequence optimal_seq;
    LevelReuseIntersectPlan optimal_reuse_plan;
    double optimal_cost;
    bool optimal_exists = false;
    //#if defined(DEBUG)
    //    size_t enumerate_count = 0;
    //#endif

    do {
      // ensure the match order is connected
      if (ConnectedMatchOrder(seq, con_)) {
        /*#if defined(DEBUG)
                std::cout << "enumerate_count=" << enumerate_count << ":";
                ++enumerate_count;
                for (size_t i = 0; i < seq.size(); ++i) {
                  if (i) std::cout << " ";
                  std::cout << seq[i];
                }
                std::cout << std::endl;
        #endif
        */
        // generate corresponding reusable plan
        LevelReuseIntersectPlan reuse_plan;
        ReusePlanGenerator::GenerateReuseTraversalPlan(
            seq, con_, avg_size_tree_node, reuse_plan);

        // estimate the cost
        double cur_cost =
            EstimateReuseTraversalCost(seq, reuse_plan, con_, avg_degree,
                                       ratio_random_access_versus_seq_access);

        if (!optimal_exists || (optimal_exists && optimal_cost > cur_cost)) {
          optimal_seq.assign(seq.begin(), seq.end());
          optimal_reuse_plan.assign(reuse_plan.begin(), reuse_plan.end());
          optimal_cost = cur_cost;
          optimal_exists = true;
        }
      }
    } while (std::next_permutation(seq.begin(), seq.end()));

    seq_.assign(optimal_seq.begin(), optimal_seq.end());
    reuse_plan_.assign(optimal_reuse_plan.begin(), optimal_reuse_plan.end());
  }

  static bool ConnectedMatchOrder(const SearchSequence& seq,
                                  const AllConnType& con) {
    size_t n = seq.size();
    std::vector<bool> vis(n, false);
    vis[seq[0]] = true;
    for (size_t i = 1; i < n; ++i) {
      auto u = seq[i];
      bool exists = false;
      for (auto nu : con[u]) {
        if (vis[nu]) {
          exists = true;
          break;
        }
      }
      if (!exists) return false;
      vis[u] = true;
    }
    return true;
  }

  // estimate the #partial instances for each level
  static void EstimatePartialInstances(const SearchSequence& seq,
                                       const AllConnType& con,
                                       const double avg_degree,
                                       std::vector<double>& partial_instances) {
    AllConnType backward_conn;
    Plan::GetBackwardConnectivity(backward_conn, con, seq);

    partial_instances.clear();
    size_t n = seq.size();
    double cur_size = 1;
    for (size_t i = 0; i < n; ++i) {
      auto u = seq[i];
      if (backward_conn[u].size() == 0) {
        assert(i == 0);
      } else {
        cur_size *= avg_degree / backward_conn.size();
      }
      partial_instances.push_back(cur_size);
    }
  }

  static double EstimateReuseTraversalCost(
      const SearchSequence& seq, const LevelReuseIntersectPlan& reuse_plan,
      const AllConnType& con, const double avg_degree,
      const double ratio_random_access_versus_seq_access) {
    std::vector<double> partial_instances;
    EstimatePartialInstances(seq, con, avg_degree, partial_instances);

    double ret = 0.0;
    size_t n = seq.size();
    for (size_t i = 0; i < n; ++i) {
      // the MATERIALIZE cost
      double Tm = 3 * partial_instances[i];
      double Tc = 0;
      if (i > 0) {
        // alpha: overhead of checking one set for all candidates
        const double bin_search_overhead = log2(avg_degree);
        double candidates;
        if (reuse_plan[i].GetReuseConnectivityMeta().size() == 0) {
          candidates = avg_degree;
        } else {
          double scale = 1;
          for (auto& reuse_conn_meta :
               reuse_plan[i].GetReuseConnectivityMeta()) {
            if (reuse_conn_meta.GetConnectivity().size() > scale) {
              scale = reuse_conn_meta.GetConnectivity().size();
            }
          }
          candidates = avg_degree / scale;
        }

        double alpha = candidates * bin_search_overhead;

        // w: the number of sets to check
        double w = reuse_plan[i].GetSeparateConnectivity().size() +
                   reuse_plan[i].GetReuseConnectivityMeta().size() - 1;

        // lambda: the cost of binary searching the cached results
        // over the tree index
        double lambda = 0;
        for (auto& reuse_conn_meta : reuse_plan[i].GetReuseConnectivityMeta()) {
          lambda += ReusePlanGenerator::TreeSearchCost(
              seq, con, seq[i], reuse_conn_meta.GetSourceVertex(),
              reuse_conn_meta.GetMapping(), avg_degree);
        }

        // the COMPUTE cost
        Tc = ratio_random_access_versus_seq_access * partial_instances[i - 1] *
             (alpha * w + lambda);
        /*#if defined(DEBUG)
                std::cout << "level=" << i << ",Tm=" << Tm << ",Tc=" << Tc
                          << ",alpha=" << alpha << ",w=" << w << ",lambda=" <<
        lambda
                          << ",#instances[i-1]=" << partial_instances[i - 1]
                          << std::endl;
        #endif
        */
      }
      ret += Tm + Tc;
    }

    return ret;
  }

 public:
  LevelReuseIntersectPlan reuse_plan_;
};

#endif
