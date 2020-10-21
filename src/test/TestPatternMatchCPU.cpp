#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "CPUGraph.h"
#include "GraphIO.h"
#include "GraphPartition.h"
#include "PreprocessGraph.h"

#include "CPUBFS.h"
#include "CPURBFS.h"
#include "CPUVF2.h"
#include "CPUWCOJ.h"

const long long ans[2][kQueryType] = {
    {2224385LL, 55107655LL, 105043837LL, 16713192LL, 16703493573LL,
     1590858494251LL, 8058169897LL, 262663639LL, 8101412570LL, 16406438768LL,
     3982336960LL, 4221802226LL, 7922128863ULL},
    {3056386LL, 468774021LL, 251755062LL, 4986965LL, 53073844144LL,
     1823152976463LL, 2943152115LL, 7211947LL, 5582737206LL, 20624052752LL,
     753156161LL, 8443803LL, 628664532ULL}};
const size_t kDefaultQuries[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

static void RunCPUInstance(int algo_type,
                           const size_t *queries = kDefaultQuries,
                           const size_t queries_num = 11) {
  const std::string filenames[] = {"../../data/com-dblp.ungraph.txt",
                                   "../../data/com-youtube.ungraph.txt"};
  int thread_num = 12;
  size_t buffer_limit = 1ULL * 1024 * 1024 * 1024 * 64;
  for (size_t file_id = 0; file_id < 2; ++file_id) {
    std::string filename = filenames[file_id];
    Graph *graph = new Graph(filename, false);
    for (size_t qidx = 0; qidx < queries_num; ++qidx) {
      size_t query_id = queries[qidx];
      assert(query_id < kQueryType);
      Query *query = new Query((QueryType)query_id);
      CPUPatternMatch *cpu = NULL;

      if (algo_type == CPU_WCOJ) {
        TraversalPlan *plan = new TraversalPlan(query, 1, 1);
        cpu = new CPUWCOJoin(plan, graph, thread_num);
        cpu->Execute();
        ASSERT_EQ(cpu->GetTotalMatchCount(), ans[file_id][query_id]);
        delete plan;
        plan = NULL;
      } else if (algo_type == CPU_VF2) {
        TraversalPlan *plan = new TraversalPlan(query, 1, 1);
        cpu = new CPUVF2(plan, graph, thread_num);
        cpu->Execute();
        ASSERT_EQ(cpu->GetTotalMatchCount(), ans[file_id][query_id]);
        delete plan;
        plan = NULL;
      } else if (algo_type == CPU_RBFS) {
        ReuseTraversalPlan *plan = new ReuseTraversalPlan(query, 1, 1);
        cpu = new CPUReuseBFS(plan, graph, thread_num, buffer_limit);
        cpu->Execute();
        ASSERT_EQ(cpu->GetTotalMatchCount(), ans[file_id][query_id]);
        delete plan;
        plan = NULL;
      } else if (algo_type == CPU_BFS) {
        ReuseTraversalPlan *plan = new ReuseTraversalPlan(query, 1, 1);
        cpu = new CPUBfs(plan, graph, thread_num, buffer_limit);
        cpu->Execute();
        ASSERT_EQ(cpu->GetTotalMatchCount(), ans[file_id][query_id]);
        delete plan;
        plan = NULL;
      } else {
        assert(false);
      }

      delete cpu;
      cpu = NULL;
      delete query;
      query = NULL;
    }
    delete graph;
    graph = NULL;
  }
}
TEST(PatternMatchCPUTest, CompactBatchWCOJTest) { RunCPUInstance(CPU_WCOJ); }

TEST(PatternMatchCPUTest, VF2Test) { RunCPUInstance(CPU_VF2); }

TEST(PatternMatchCPUTest, RBFSTest) {
  const size_t queries[] = {2, 3, 6, 7, 9, 10, 11, 12};
  // const size_t queries_num = 9;
  const size_t queries_num = 8;
  RunCPUInstance(CPU_RBFS, queries, queries_num);
}
TEST(PatternMatchCPUTest, CompactBatchBFSTest) {
  const size_t queries[] = {2, 3, 6, 7, 9, 10, 11, 12};
  // const size_t queries_num = 9;
  const size_t queries_num = 8;
  RunCPUInstance(CPU_BFS, queries, queries_num);
}
TEST(ReuseBfsTest, CompactBatchQ5SimpleGraph) {
  const int N = 20;
  std::vector<std::vector<uintV>> graph_data(N);
  const size_t vertex_count = N;
  for (size_t u = 0; u < N; ++u) {
    for (size_t v = 0; v < N; ++v) {
      if (u != v) {
        graph_data[u].push_back(v);
      }
    }
  }
  Graph *graph = new Graph(graph_data);
  size_t thread_num = 12;
  size_t buffer_limit = 1ULL * 1024 * 1024 * 1024 * 64;
  Query *query = new Query(Q5);
  ReuseTraversalPlan *plan = new ReuseTraversalPlan(query, 1, 1);
  size_t exp_count = 0;

  {
    CPUWCOJoin *cpu = new CPUWCOJoin(plan, graph, thread_num);
    cpu->Execute();
    exp_count = cpu->GetTotalMatchCount();
    delete cpu;
    cpu = NULL;
  }

  {
    CPUReuseBFS *cpu = new CPUReuseBFS(plan, graph, thread_num, buffer_limit);
    cpu->Execute();
    size_t actual_count = cpu->GetTotalMatchCount();
    ASSERT_EQ(actual_count, exp_count);
    delete cpu;
    cpu = NULL;
  }

  {
    CPUBfs *cpu = new CPUBfs(plan, graph, thread_num, buffer_limit);
    cpu->Execute();
    size_t actual_count = cpu->GetTotalMatchCount();
    ASSERT_EQ(actual_count, exp_count);
    delete cpu;
    cpu = NULL;
  }

  delete plan;
  plan = NULL;
  delete query;
  query = NULL;
  delete graph;
  graph = NULL;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
