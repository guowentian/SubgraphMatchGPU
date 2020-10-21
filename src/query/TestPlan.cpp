#include "CommandLine.h"

#include "LazyTraversalPlan.h"
#include "ReuseTraversalPlan.h"

#include <iostream>

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cout << "./test_plan -a INTRA_PARTITION -t PLAN_TYPE -q QUERY -m MODE "
                 "-o VARIANT"
              << std::endl;
    std::cout << "-a: 0 for INTRA_PARTITION, 1 for INTRA_PARTITION and "
                 "INTER-PARTITION"
              << std::endl;
    std::cout << "-t: 0 for TRAVERSAL_PLAN, 1 for REUSE_TRAVERSAL_PLAN, 2 for "
                 "LAZY_TRAVERSAL_PLAN"
              << std::endl;
    return -1;
  }

  CommandLine cmd(argc, argv);
  int workload_type = cmd.GetOptionIntValue("-a", 0);
  int plan_type = cmd.GetOptionIntValue("-t", 1);
  int query_type = cmd.GetOptionIntValue("-q", 0);
  ExecuteMode execute_mode = (ExecuteMode)cmd.GetOptionIntValue("-m", 1);
  Variant variant = (Variant)cmd.GetOptionIntValue("-o", 0);
  Query* query = new Query((QueryType)query_type);

  std::cout << "workload_type=" << workload_type << ",plan_type=" << plan_type
            << ",query_type=" << query_type
            << ",execute_mode=" << (int)execute_mode
            << ",variant=" << (int)variant << std::endl;

  Plan* plan = NULL;
  if (plan_type == 0) {
    plan = new TraversalPlan(query, 1, 1, execute_mode, variant);
  } else if (plan_type == 1) {
    plan = new ReuseTraversalPlan(query, 1, 1, execute_mode, variant, true);
  } else if (plan_type == 2) {
    plan = new LazyTraversalPlan(
        query, 1, 1, execute_mode, variant,
        LazyTraversalCompressLevel::COMPRESS_LEVEL_NON_MATERIALIZE_OPT);
  }
  if (workload_type == 1) {
    plan->OptimizePlan();
  }
  plan->Print();

  return 0;
}
