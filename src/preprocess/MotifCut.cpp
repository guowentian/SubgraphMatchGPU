#include "MotifCut.h"
#include "CommandLine.h"
#include "Query.h"
#include "SampleMotifFinder.h"
#include "TraversalPlan.h"

int main(int argc, char *argv[]) {
  if (argc == 1) {
    std::cout
        << "motifcut -a finder_type -f filename -d directed -p "
           "partition_type -t thread_num -o output_filename -z output_type"
        << std::endl;
    std::cout << "PARTITION_TYPE: "
              << "Q0 (TRIANGLE) " << Q0 << ", Q1 (square) " << Q1
              << ", Q2 (chordal square) " << Q2 << ", Q3 (4 clique) " << Q3
              << ", Q4 (house) " << Q4 << ", Q5 (quad triangle) " << Q5
              << ", Q6 (near5clique) " << Q6 << ", Q7 (5 clique) " << Q7
              << ", Q8 (chordal roof) " << Q8 << ", Q9 (three triangle) " << Q9
              << ", Q10 (solar square) " << Q10 << ", Q11 (6 clique) " << Q11
              << ",Q" << LINE << " (LINE, Default)" << std::endl;

    std::cout << "FINDER_TYPE: 0 for exact (Default), 1 for sample"
              << std::endl;
    std::cout << "OUTPUT_TYPE: 0 for MOTIF_CUT, 1 for METIS_WEIGHT, 2 for "
                 "METIS_VE_WEIGHT"
              << std::endl;
    return -1;
  }
  CommandLine cmd(argc, argv);
  std::string filename =
      cmd.GetOptionValue("-f", "../../data/com-dblp.ungraph.txt");
  int directed = cmd.GetOptionIntValue("-d", 0);
  int partition_type = cmd.GetOptionIntValue("-p", LINE);
  int thread_num = cmd.GetOptionIntValue("-t", 1);
  int finder_type = cmd.GetOptionIntValue("-a", 0);
  std::string output_filename = cmd.GetOptionValue("-o", "");
  int motifcut_output_type = cmd.GetOptionIntValue("-z", 1);

  std::cout << "filename=" << filename << ",directed=" << directed
            << ",partition_type=" << partition_type
            << ",thread_num=" << thread_num
            << ",output_filename=" << output_filename
            << ",motifcut_output_type=" << motifcut_output_type << std::endl;

  Graph *graph = new Graph(filename, directed);
  Query *query = new Query((QueryType)partition_type);
  TraversalPlan *query_plan = new TraversalPlan(query, 1, 1);
  query_plan->Print();

  MotifFinder *motif_finder = NULL;
  if (finder_type == 0) {
    motif_finder = new MotifFinder(query_plan, graph, thread_num);
  } else {
    motif_finder = new SampleMotifFinder(query_plan, graph, thread_num);
  }
  MotifCut *motifcut = new MotifCut(
      filename, graph, partition_type, motif_finder, output_filename,
      (MotifCut::MotifCultOutputType)motifcut_output_type);
  motifcut->Execute();
  std::cout << "done!" << std::endl;

  return 0;
}
