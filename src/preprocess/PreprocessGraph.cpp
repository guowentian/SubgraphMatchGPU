#include "PreprocessGraph.h"
#include <iostream>
#include "CommandLine.h"

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cout << "./preprocess_graph -f GRAPH_FILENAME(SNAP format) -d "
                 "DIRECTED -o OUTPUT_FORMAT"
              << std::endl;
    std::cout << "OUTPUT_FORMAT: " << GraphFormat::SNAP << "  for .snap, "
              << GraphFormat::BIN << " (Default) for .bin, " << GraphFormat::ADJ
              << " for .adj, " << std::endl;
    return -1;
  }

  CommandLine cmd(argc, argv);
  std::string filename =
      cmd.GetOptionValue("-f", "../../data/com-dblp.ungraph.txt");
  int directed = cmd.GetOptionIntValue("-d", 0);
  int output_format = cmd.GetOptionIntValue("-o", GraphFormat::BIN);

  Graph* graph = new Graph(filename, directed);
  PreprocessGraph* preprocess = new PreprocessGraph(graph);
  preprocess->Preprocess();

  GraphIO::Validate(graph->GetVertexCount(), graph->GetRowPtrs(),
                    graph->GetCols());
  GraphIO::WriteDataFile((GraphFormat)output_format, filename, directed,
                         graph->GetVertexCount(), graph->GetEdgeCount(),
                         graph->GetRowPtrs(), graph->GetCols());

  delete preprocess;
  preprocess = NULL;
  delete graph;
  graph = NULL;

  return 0;
}
