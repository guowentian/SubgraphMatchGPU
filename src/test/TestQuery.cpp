#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <unordered_map>

#include "LazyTraversalPlan.h"
#include "ReusePlanGenerator.h"
#include "ReuseTraversalOptimizer.h"
#include "TraversalPlanGenerator.h"

#include "TraversalPlan.h"

static void GenerateQ2(AllConnType& conn) {
  size_t vertex_count = 4;
  conn.resize(vertex_count);
  conn[0].push_back(1);
  conn[0].push_back(2);
  conn[0].push_back(3);
  conn[1].push_back(0);
  conn[1].push_back(2);
  conn[2].push_back(0);
  conn[2].push_back(1);
  conn[2].push_back(3);
  conn[3].push_back(0);
  conn[3].push_back(2);
}

static void GenerateQ5(AllConnType& conn) {
  size_t vertex_count = 6;
  conn.resize(vertex_count);
  conn[0].push_back(1);
  conn[0].push_back(2);
  conn[0].push_back(3);
  conn[0].push_back(4);
  conn[0].push_back(5);
  conn[1].push_back(0);
  conn[1].push_back(2);
  conn[2].push_back(0);
  conn[2].push_back(1);
  conn[2].push_back(3);
  conn[3].push_back(0);
  conn[3].push_back(2);
  conn[3].push_back(4);
  conn[4].push_back(0);
  conn[4].push_back(3);
  conn[4].push_back(5);
  conn[5].push_back(0);
  conn[5].push_back(4);
}

static void Print(LevelReuseIntersectPlan& level_reuse_intersect_plan,
                  const SearchSequence& seq) {
  size_t vertex_count = seq.size();
  for (size_t i = 0; i < vertex_count; ++i) {
    std::cout << "level " << i << ", vertex=" << seq[i] << ":";
    level_reuse_intersect_plan[i].Print();
    std::cout << std::endl;
  }
}

TEST(ReusePlan, GenerateReuseTraversalVertexPlan) {
  AllConnType conn;
  GenerateQ2(conn);
  const uintV seq_array[] = {0, 2, 1, 3};
  SearchSequence seq(seq_array, seq_array + 4);

  std::vector<VertexMapping> mappings;
  ReusePlanGenerator::Subsume(seq, conn, 3, 1, mappings);
  ASSERT_EQ(mappings.size(), 1);
  auto& mp = mappings[0];
  ASSERT_EQ(mp[0], 0);
  ASSERT_EQ(mp[2], 2);
  ASSERT_EQ(mp[1], 3);

  VertexReuseIntersectPlan reuse_vertex_plan;
  ReusePlanGenerator::GenerateReuseTraversalVertexPlan(seq, conn, 3, 12,
                                                       reuse_vertex_plan);

  ASSERT_EQ(reuse_vertex_plan.GetSeparateConnectivity().size(), 0);
  ASSERT_EQ(reuse_vertex_plan.GetReuseConnectivityMeta().size(), 1);

  auto& reuse_conn_meta = reuse_vertex_plan.GetReuseConnectivityMeta()[0];
  ASSERT_EQ(reuse_conn_meta.mapping_[0], 0);
  ASSERT_EQ(reuse_conn_meta.mapping_[2], 2);
  ASSERT_EQ(reuse_conn_meta.mapping_[1], 3);
}

TEST(ReusePlan, ReuseTraversalVertexPlanSpecificOrderCompareQ5) {
  AllConnType conn;
  GenerateQ5(conn);
  const uintV seq_array1[] = {0, 3, 2, 1, 4, 5};
  SearchSequence seq1(seq_array1, seq_array1 + 6);
  const uintV seq_array2[] = {0, 4, 3, 2, 1, 5};
  SearchSequence seq2(seq_array2, seq_array2 + 6);

  std::vector<VertexReuseIntersectPlan> reuse_vertex_plan1;
  ReusePlanGenerator::GenerateReuseTraversalPlan(seq1, conn, 12,
                                                 reuse_vertex_plan1);

  std::vector<VertexReuseIntersectPlan> reuse_vertex_plan2;
  ReusePlanGenerator::GenerateReuseTraversalPlan(seq2, conn, 12,
                                                 reuse_vertex_plan2);

  Print(reuse_vertex_plan1, seq1);
  double cost1 = ReuseTraversalOptimizer::EstimateReuseTraversalCost(
      seq1, reuse_vertex_plan1, conn, 12, 10);
  std::cout << "cost1=" << cost1 << std::endl;

  Print(reuse_vertex_plan2, seq2);
  double cost2 = ReuseTraversalOptimizer::EstimateReuseTraversalCost(
      seq2, reuse_vertex_plan2, conn, 12, 10);
  std::cout << "cost2=" << cost2 << std::endl;
}

static void VerifyUniqueSequence(SearchSequence& seq) {
  size_t n = seq.size();
  std::vector<bool> vis(n, false);
  for (size_t i = 0; i < n; ++i) {
    ASSERT_EQ(seq[i] < n, true);
    ASSERT_EQ(vis[seq[i]], false);
    vis[seq[i]] = true;
  }
}

static void SameConnectivity(const AllConnType& graph,
                             const AllConnType& pattern) {
  ASSERT_EQ(graph.size(), pattern.size());
  for (size_t u = 0; u < graph.size(); ++u) {
    ASSERT_EQ(graph[u].size(), pattern[u].size());
    for (auto v : graph[u]) {
      bool exists = false;
      for (auto v2 : pattern[u]) {
        if (v2 == v) {
          exists = true;
        }
      }
      ASSERT_EQ(exists, true);
    }
  }
}

// extract the vertex id < bound
static void ExtractSubGraph(AllConnType& ret, const AllConnType& conn,
                            size_t bound) {
  ret.clear();
  size_t n = conn.size();
  ret.resize(n);
  for (size_t i = 0; i < n; ++i) {
    if (i < bound) {
      for (auto v : conn[i]) {
        if (v < bound) {
          ret[i].push_back(v);
        }
      }
    }
  }
}

TEST(InterPartitionTest, TraversalPlan) {
  ExecuteMode mode = HYBRID_CPU_GPU;
  Variant variant = O2;
  for (size_t query_id = 0; query_id < kQueryType; ++query_id) {
    Query* q = new Query((QueryType)query_id);
    TraversalPlan* plan = new TraversalPlan(q, 1, 1, mode, variant);
    plan->OptimizePlan();
    std::cout << q->GetString() << std::endl;
    plan->Print();

    // load data
    std::vector<SearchSequence> dfs_orders;
    MultiDfsIdGroup group_dfs_ids;
    LevelMultiDfsIdGroup level_group_dfs_ids;
    plan->LoadSearchSequences(dfs_orders);
    plan->LoadMultiDfsIdGroup(group_dfs_ids);
    plan->LoadLevelMultiDfsIdGroup(level_group_dfs_ids);
    size_t vertex_count = plan->GetVertexCount();
    auto& conn = plan->GetConnectivity();

    // uniqueness of each single sequence
    for (size_t i = 0; i < dfs_orders.size(); ++i) {
      VerifyUniqueSequence(dfs_orders[i]);
    }

    // covered all edges
    std::vector<std::vector<bool>> edge_vis(vertex_count);
    for (size_t i = 0; i < vertex_count; ++i) {
      edge_vis[i].resize(vertex_count, false);
    }
    size_t covered_edge_count = 0;
    for (size_t i = 0; i < dfs_orders.size(); ++i) {
      auto u1 = dfs_orders[i][0];
      auto u2 = dfs_orders[i][1];
      ASSERT_EQ(edge_vis[u1][u2], false);
      ASSERT_EQ(edge_vis[u2][u1], false);

      edge_vis[u1][u2] = true;
      edge_vis[u2][u1] = true;
      ++covered_edge_count;
    }
    size_t exp_covered_edge_count = 0;
    for (auto& adj : conn) {
      exp_covered_edge_count += adj.size();
    }
    ASSERT_EQ(covered_edge_count, exp_covered_edge_count / 2);

    // prefix equivalence
    for (size_t level = 0; level < level_group_dfs_ids.size(); ++level) {
      auto& dfs_groups = level_group_dfs_ids[level];
      for (auto& group : dfs_groups) {
        size_t first_dfs_id = group[0];
        AllConnType pattern_intersect_indices;
        TraversalPlanGenerateHelper::GetIndexBasedConnectivity(
            pattern_intersect_indices, conn, dfs_orders[first_dfs_id],
            vertex_count);
        AllConnType pattern_prefix_intersect_indices;
        ExtractSubGraph(pattern_prefix_intersect_indices,
                        pattern_intersect_indices, level + 1);

        for (size_t group_index = 1; group_index < group.size();
             ++group_index) {
          size_t dfs_id = group[group_index];
          AllConnType cur_intersect_indices;
          TraversalPlanGenerateHelper::GetIndexBasedConnectivity(
              cur_intersect_indices, conn, dfs_orders[dfs_id], vertex_count);
          AllConnType cur_prefix_intersect_indices;
          ExtractSubGraph(cur_prefix_intersect_indices, cur_intersect_indices,
                          level + 1);

          SameConnectivity(cur_prefix_intersect_indices,
                           pattern_prefix_intersect_indices);
        }
      }
    }

    delete q;
    q = NULL;
    delete plan;
    plan = NULL;
  }
}

static void VerifyLazyTraversalIndexBasedExecuteSequence(
    std::vector<LazyTraversalEntry>& exec_seq, const SearchSequence& seq,
    const AllConnType& conn, size_t vertex_count) {
  AllConnType backward_conn;
  Plan::GetOrderedIndexBasedConnectivity(backward_conn, conn, seq);

  std::vector<bool> materialized(vertex_count, false);
  for (auto entry : exec_seq) {
    if (entry.first == MATERIALIZE) {
      ASSERT_EQ(materialized[entry.second], false);
      materialized[entry.second] = true;
    }
    if (entry.first == COMPUTE || entry.first == COMPUTE_COUNT ||
        entry.first == COMPUTE_PATH_COUNT) {
      auto u = entry.second;
      for (auto prev_u : backward_conn[u]) {
        ASSERT_EQ(materialized[prev_u], true);
      }
    }
  }
}

TEST(InterPartitionTest, LazyTraversalPlan) {
  ExecuteMode mode = HYBRID_CPU_GPU;
  Variant variant = O2;
  for (size_t query_id = 0; query_id < kQueryType; ++query_id) {
    Query* q = new Query((QueryType)query_id);
    LazyTraversalPlan* plan = new LazyTraversalPlan(q, 1, 1, mode, variant);
    plan->OptimizePlan();
    std::cout << q->GetString() << std::endl;
    plan->Print();

    // load data
    std::vector<SearchSequence> dfs_orders;
    MultiDfsIdGroup group_dfs_ids;
    LevelMultiDfsIdGroup level_group_dfs_ids;
    plan->LoadSearchSequences(dfs_orders);
    plan->LoadMultiDfsIdGroup(group_dfs_ids);
    plan->LoadLevelMultiDfsIdGroup(level_group_dfs_ids);
    auto vertex_count = plan->GetVertexCount();
    auto& conn = plan->GetConnectivity();

    auto& inter_part_exec_seq = plan->GetInterPartitionExecuteOperations();

    // each execute sequence is valid in terms of connectivity
    for (size_t dfs_id = 0; dfs_id < dfs_orders.size(); ++dfs_id) {
      VerifyLazyTraversalIndexBasedExecuteSequence(
          inter_part_exec_seq[dfs_id], dfs_orders[dfs_id], conn, vertex_count);
    }

    // topo-equivalent search sequences should have the same execute sequence
    for (size_t group_id = 0; group_id < group_dfs_ids.size(); ++group_id) {
      for (size_t i = 1; i < group_dfs_ids[group_id].size(); ++i) {
        size_t first_dfs_id = group_dfs_ids[group_id][0];
        size_t second_dfs_id = group_dfs_ids[group_id][i];

        ASSERT_EQ(inter_part_exec_seq[first_dfs_id].size(),
                  inter_part_exec_seq[second_dfs_id].size());
        for (size_t j = 0; j < inter_part_exec_seq[first_dfs_id].size(); ++j) {
          ASSERT_EQ(inter_part_exec_seq[first_dfs_id][j],
                    inter_part_exec_seq[second_dfs_id][j]);
        }
      }
    }

    delete q;
    q = NULL;
    delete plan;
    plan = NULL;
  }
}

TEST(QueryTest, DISABLED_TraversalPlanTest) {
  ExecuteMode mode = HYBRID_CPU_GPU;
  Variant variant = O2;
  for (size_t query_id = 0; query_id < kQueryType; ++query_id) {
    Query* q = new Query((QueryType)query_id);
    TraversalPlan* plan = new TraversalPlan(q, 1, 1, mode, variant);
    plan->OptimizePlan();
    std::cout << q->GetString() << std::endl;
    plan->Print();
    delete q;
    q = NULL;
    delete plan;
    plan = NULL;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
