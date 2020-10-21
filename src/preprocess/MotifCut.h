#ifndef __MOTIF_CUT_H__
#define __MOTIF_CUT_H__

#include "CPUGraph.h"
#include "Meta.h"
#include "MotifFinder.h"
#include "ParallelUtils.h"

#include <algorithm>
#include <cstring>
#include <fstream>

class MotifCut {
 public:
  enum MotifCultOutputType { MOTIF_CUT, METIS_WEIGHT, METIS_VE_WEIGHT };

  MotifCut(std::string full_filename, Graph *rel, size_t partition_type,
           MotifFinder *motif_finder, std::string output_filename,
           MotifCultOutputType output_type)
      : full_filename_(full_filename),
        output_filename_(output_filename),
        graph_(rel),
        partition_type_(partition_type),
        motif_finder_(motif_finder),
        output_type_(output_type) {
    weights_ = new WeightType[graph_->GetEdgeCount()];
    memset(weights_, 0, sizeof(WeightType) * graph_->GetEdgeCount());
  }
  ~MotifCut() {
    delete[] weights_;
    weights_ = NULL;
    delete motif_finder_;
    motif_finder_ = NULL;
  }
  void Execute() {
    if (partition_type_ == LINE) {
      Uniform();
    } else {
      motif_finder_->Execute();
      size_t *edge_cross_times = motif_finder_->GetEdgeCrossTimes();
      parallel_for(size_t i = 0; i < graph_->GetEdgeCount(); ++i) {
        weights_[i] = edge_cross_times[i] + 1;
      }
    }

    SamplePrintWeights();

    std::string output_filename = output_filename_;
    // if the output filename is not specified, generate a filename based on the
    // execution property
    if (output_filename == "") {
      output_filename =
          GenerateOutputFileName(full_filename_, partition_type_, output_type_);
    }

    switch (output_type_) {
      case MOTIF_CUT:
        WriteMotifCutWeightFile(output_filename);
        break;
      case METIS_WEIGHT:
        WriteMetisFile(output_filename);
        break;
      case METIS_VE_WEIGHT:
        WriteMetisVertexWeightFile(output_filename);
        break;
      default:
        break;
    }
  }

  static std::string GenerateOutputFileName(std::string full_filename,
                                            size_t partition_type,
                                            MotifCultOutputType output_type) {
    std::string property = "";
    if (partition_type == LINE) {
      property = "uniform";
    } else {
      if (partition_type == Q0) {
        property = "triangle";
      } else if (partition_type == Q1) {
        property = "square";
      } else if (partition_type == Q2) {
        property = "chordalsquare";
      } else if (partition_type == Q3) {
        property = "fourclique";
      } else if (partition_type == Q4) {
        property = "house";
      } else if (partition_type == Q5) {
        property = "quadsquare";
      } else if (partition_type == Q6) {
        property = "near5clique";
      } else if (partition_type == Q7) {
        property = "fiveclique";
      } else if (partition_type == Q8) {
        property = "q8";
      } else if (partition_type == Q9) {
        property = "q9";
      } else if (partition_type == Q10) {
        property = "q10";
      } else if (partition_type == Q11) {
        property = "q11";
      } else {
        assert(false);
      }
    }

    std::string output_property = "";
    if (output_type == MOTIF_CUT) {
      output_property = ".motifcut.";
    } else if (output_type == METIS_WEIGHT) {
      output_property = ".weight.";
    } else if (output_type == METIS_VE_WEIGHT) {
      output_property = ".veweight.";
    } else {
      assert(false);
    }

    return full_filename + output_property + property;
  }

  void SamplePrintWeights() {
    for (size_t i = 0; i < 10; ++i) {
      std::cout << " " << weights_[i];
    }
    std::cout << std::endl;
  }
  void Uniform() {
    for (size_t i = 0; i < graph_->GetEdgeCount(); ++i) {
      weights_[i] = 1;
    }
  }

  // This is an output file format required by yuchen.
  // It records the weight for each edge. The weight is the number of motifs
  // containing that edge.
  void WriteMotifCutWeightFile(std::string filename) {
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();

    std::cout << "write to file " << filename << std::endl;
    std::ofstream file(filename.c_str(), std::fstream::out);
    // first row is total number of motifs
    size_t total_match_count = motif_finder_->GetTotalMatchCount();
    file << total_match_count << std::endl;

    size_t *edge_cross_times = motif_finder_->GetEdgeCrossTimes();
    for (size_t u = 0; u < graph_->GetVertexCount(); ++u) {
      for (size_t j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        // for each line: u v w
        auto v = cols[j];
        WeightType w = edge_cross_times[j];
        file << u << " " << v << " " << w << std::endl;
        assert(v < graph_->GetVertexCount() && v >= 0);
      }
    }

    file.close();
  }

  // As Metis program accepts a different format of graph files, which is
  // adjacent lists with the weight for each edge,
  // we write the file with the required format here.
  void WriteMetisFile(std::string filename) {
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();

    std::cout << "write to file " << filename << std::endl;
    std::ofstream file(filename.c_str(), std::fstream::out);
    // metis only counts edge between of any pair only once
    file << graph_->GetVertexCount() << " "
         << (graph_->GetDirected() ? graph_->GetEdgeCount()
                                   : graph_->GetEdgeCount() / 2)
         << " "
         << "001" << std::endl;

    for (size_t u = 0; u < graph_->GetVertexCount(); ++u) {
      bool first = true;
      for (size_t j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        auto v = cols[j];
        WeightType w = weights_[j];
        if (!first) file << " ";
        // metis is 1-based
        file << v + 1 << " " << w;
        first = false;
        assert(v < graph_->GetVertexCount() && v >= 0);
      }
      file << std::endl;
    }

    file.close();
  }

  // As METIS partitions by vertices instead of edges, the resulting partitions
  // may have similar number of vertices but quite different number of edges.
  // This causes imbalanced size of different partitions.
  // To avoid this issue, our heuristic is to assign weights to vertices.
  void WriteMetisVertexWeightFile(std::string filename) {
    auto row_ptrs = graph_->GetRowPtrs();
    auto cols = graph_->GetCols();

    std::cout << "write to file " << filename << std::endl;
    std::ofstream file(filename.c_str(), std::fstream::out);
    // metis only counts edge between of any pair only once
    file << graph_->GetVertexCount() << " "
         << (graph_->GetDirected() ? graph_->GetEdgeCount()
                                   : graph_->GetEdgeCount() / 2)
         << " "
         << "011 1" << std::endl;

    for (size_t u = 0; u < graph_->GetVertexCount(); ++u) {
      auto vertex_weight = row_ptrs[u + 1] - row_ptrs[u];
      file << vertex_weight;
      for (size_t j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        auto v = cols[j];
        WeightType w = weights_[j];
        // metis is 1-based
        file << " " << v + 1 << " " << w;
        assert(v < graph_->GetVertexCount() && v >= 0);
      }
      file << std::endl;
    }

    file.close();
  }

 public:
  std::string full_filename_;
  std::string output_filename_;

  Graph *graph_;
  size_t partition_type_;
  WeightType *weights_;
  MotifFinder *motif_finder_;
  MotifCultOutputType output_type_;
};

#endif
