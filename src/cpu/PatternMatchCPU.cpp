#include <iostream>

#include "CPUBFS.h"
#include "CPURBFS.h"
#include "CPUVF2.h"
#include "CPUWCOJ.h"
#include "Meta.h"

#include "CommandLine.h"

static void HelperMsg(int argc, char *argv[]) {
  if (argc == 1) {
    std::cout << "./patternmatchcpu -f FILENAME -d IS_DIRECTRED -a ALGORITHM "
                 "-q QUERY "
                 "-t THREAD_NUM -b BUFFER_LIMIT(GB)"
              << std::endl;
    std::cout << "ALGORITHM: "
              << "CPU_WCOJ=" << CPU_WCOJ << ", CPU_VF2=" << CPU_VF2
              << ", CPU_RBFS=" << CPU_RBFS << ", CPU_BFS=" << CPU_BFS
              << std::endl;
    std::cout << "QUERY: "
              << "Q0 (TRIANGLE) " << Q0 << ", Q1 (square) " << Q1
              << ", Q2 (chordal square) " << Q2 << ", Q3 (4 clique) " << Q3
              << ", Q4 (house) " << Q4 << ", Q5 (quad triangle) " << Q5
              << ", Q6 (near5clique) " << Q6 << ", Q7 (5 clique) " << Q7
              << ", Q8 (chordal roof) " << Q8 << ", Q9 (three triangle) " << Q9
              << ", Q10 (solar square) " << Q10 << ", Q11 (6 clique) " << Q11
              << std::endl;
    exit(-1);
  }
}
int main(int argc, char *argv[]) {
  HelperMsg(argc, argv);

  CommandLine cmd(argc, argv);
  std::string filename =
      cmd.GetOptionValue("-f", "../../data/com-dblp.ungraph.txt");
  int directed = cmd.GetOptionIntValue("-d", 1);
  int algo = cmd.GetOptionIntValue("-a", CPU_WCOJ);
  int query_type = cmd.GetOptionIntValue("-q", Q0);
  std::string partition_filename = cmd.GetOptionValue("-e", "");
  int partition_num = cmd.GetOptionIntValue("-p", 1);
  int thread_num = cmd.GetOptionIntValue("-t", 1);
  int dev_num = cmd.GetOptionIntValue("-v", 1);
  size_t buffer_limit_gb = cmd.GetOptionIntValue("-b", 64);
  size_t buffer_limit = buffer_limit_gb * 1024 * 1024 * 1024;
  std::cout << "filename=" << filename << ",directed=" << directed
            << ",algo=" << algo << ",query_type=" << query_type << std::endl;
  std::cout << ",partition_filename=" << partition_filename
            << ",partition_num=" << partition_num
            << ",thread_num=" << thread_num << ",buffer_limit=" << buffer_limit
            << std::endl;

  Query *query = new Query((QueryType)query_type);
  Graph *graph = new Graph(filename, directed);

  if (algo == CPU_WCOJ) {
    TraversalPlan *query_plan =
        new TraversalPlan(query, partition_num, dev_num);
    query_plan->Print();

    CPUWCOJoin *dfs_match = new CPUWCOJoin(query_plan, graph, thread_num);
    dfs_match->Execute();
  } else if (algo == CPU_VF2) {
    TraversalPlan *query_plan =
        new TraversalPlan(query, partition_num, dev_num);
    query_plan->Print();

    CPUVF2 *cpu_vf2 = new CPUVF2(query_plan, graph, thread_num);
    cpu_vf2->Execute();
  } else if (algo == CPU_RBFS) {
    ReuseTraversalPlan *plan =
        new ReuseTraversalPlan(query, partition_num, dev_num);
    plan->Print();

    CPUReuseBFS *rbfs = new CPUReuseBFS(plan, graph, thread_num, buffer_limit);
    rbfs->Execute();
  } else if (algo == CPU_BFS) {
    ReuseTraversalPlan *plan =
        new ReuseTraversalPlan(query, partition_num, dev_num);
    plan->Print();

    CPUBfs *bfs = new CPUBfs(plan, graph, thread_num, buffer_limit);
    bfs->Execute();
  } else {
    assert(false);
  }

  return 0;
}
