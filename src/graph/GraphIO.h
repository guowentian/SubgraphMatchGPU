#ifndef __GRAPH_IO_H__
#define __GRAPH_IO_H__

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

#include "Meta.h"
#include "TimeMeasurer.h"
#if defined(OPENMP)
#include <omp.h>
#endif

enum GraphFormat { SNAP, BIN, SNAP_LVID, ADJ };

class GraphIO {
 public:
  static void ReadDataFile(std::string& filename, bool directed,
                           size_t& vertex_count, size_t& edge_count,
                           uintE*& row_ptrs, uintV*& cols,
                           GraphFormat graph_format) {
    switch (graph_format) {
      case SNAP:
        ReadSNAPFile(filename, directed, vertex_count, edge_count, row_ptrs,
                     cols);
        break;
      case BIN:
        ReadCSRBinFile(filename, directed, vertex_count, edge_count, row_ptrs,
                       cols);
        break;
      case SNAP_LVID:
        ReadSNAPLargeVIDFile(filename, directed, vertex_count, edge_count,
                             row_ptrs, cols);
        break;
      default:
        assert(false);
    }
  }

  static void ReadDataFile(std::string& filename, bool directed,
                           size_t& vertex_count, size_t& edge_count,
                           uintE*& row_ptrs, uintV*& cols) {
    std::string suffix = filename.substr(filename.rfind(".") + 1);
    GraphFormat graph_format;
    if (suffix == "txt") {
      graph_format = SNAP;
    } else if (suffix == "bin") {
      graph_format = BIN;
    } else if (suffix == "lvid") {
      graph_format = SNAP_LVID;
    } else {
      assert(false);
    }
    ReadDataFile(filename, directed, vertex_count, edge_count, row_ptrs, cols,
                 graph_format);
  }

  static void ReadPartitionFile(std::string& filename, size_t partition_num,
                                size_t vertex_count,
                                uintP* vertex_partition_map) {
    if (partition_num == 1 || filename.length() == 0) {
      partition_num = 1;
      for (uintV u = 0; u < vertex_count; u++) {
        vertex_partition_map[u] = 0;
      }
    } else {
      std::ifstream file(filename.c_str(), std::fstream::in);
      size_t pid;
      size_t max_pid = 0;
      for (uintV u = 0; u < vertex_count; ++u) {
        file >> pid;
        max_pid = std::max(max_pid, pid);
        vertex_partition_map[u] = pid;
      }
      file.close();
      assert(max_pid + 1 == partition_num);
    }
  }

  static void ReadCSRBinFile(std::string& filename, bool directed,
                             size_t& vertex_count, size_t& edge_count,
                             uintE*& row_ptrs, uintV*& cols) {
    // CSR is a homemade format
    // It removes parallel edges and self loops, but it may contain dangling
    // nodes
    vertex_count = 0;
    edge_count = 0;
    row_ptrs = NULL;
    cols = NULL;

    TimeMeasurer timer;
    timer.StartTimer();
    std::cout << "start read csr bin file....";
    FILE* file_in = fopen(filename.c_str(), "rb");
    assert(file_in != NULL);
    fseek(file_in, 0, SEEK_SET);
    size_t res = 0;
    size_t uintV_size = 0, uintE_size = 0;
    res += fread(&uintV_size, sizeof(size_t), 1, file_in);
    res += fread(&uintE_size, sizeof(size_t), 1, file_in);
    res += fread(&vertex_count, sizeof(size_t), 1, file_in);
    res += fread(&edge_count, sizeof(size_t), 1, file_in);
    std::cout << "uintV_size=" << uintV_size << ",uintE_size=" << uintE_size
              << std::endl;
    std::cout << "vertex_count=" << vertex_count << ",edge_count=" << edge_count
              << std::endl;
    assert(uintV_size == sizeof(uintV));
    assert(uintE_size == sizeof(uintE));

    row_ptrs = new uintE[vertex_count + 1];
    cols = new uintV[edge_count];
    for (uintV u = 0; u <= vertex_count; ++u) {
      res += fread(&row_ptrs[u], sizeof(uintE), 1, file_in);
    }
    for (uintV u = 0; u < vertex_count; ++u) {
      for (uintE j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        res += fread(&cols[j], sizeof(uintV), 1, file_in);
      }
    }
    assert(res == (4 + (vertex_count + 1) + edge_count));
    fgetc(file_in);
    assert(feof(file_in));
    fclose(file_in);

    timer.EndTimer();
    std::cout << "finish read csr bin file, elapsed_time="
              << timer.GetElapsedMicroSeconds() / 1000.0 << "ms" << std::endl;
  }

  // Some data graphs may contain large vertex ids, e.g., in web graphs,
  // the vertex ids could be some large integers that cannot be stored in
  // even in long long int.
  // So we need to use a map to sequentialize the vertex ids.
  static void ReadSNAPLargeVIDFile(std::string& filename, bool directed,
                                   size_t& vertex_count, size_t& edge_count,
                                   uintE*& row_ptrs, uintV*& cols) {
    vertex_count = 0;
    edge_count = 0;
    row_ptrs = NULL;
    cols = NULL;

    std::cout << "start build csr..." << std::endl;
    // const char* kDelimiters = " ,;\t";
    const char* kDelimiters = "0123456789";
    std::unordered_map<std::string, uintV> ids;
    std::vector<uintV> edge_pairs;
    {
      std::ifstream file(filename.c_str(), std::fstream::in);
      std::string line;
      while (getline(file, line)) {
        if (line.length() == 0 || !std::isdigit(line[0])) continue;

        std::vector<std::string> num_strs;
        size_t cur_pos = 0;
        while (cur_pos < line.length()) {
          cur_pos =
              line.find_first_of(kDelimiters, cur_pos, strlen(kDelimiters));
          if (cur_pos < line.length()) {
            size_t next_pos = line.find_first_not_of(kDelimiters, cur_pos,
                                                     strlen(kDelimiters));
            num_strs.push_back(line.substr(cur_pos, next_pos - cur_pos));
            assert(next_pos > cur_pos);
            cur_pos = next_pos;
          }
        }

        for (auto& str : num_strs) {
          assert(str.length());
          for (auto ch : str) {
            assert(std::isdigit(ch));
          }
        }

        for (auto& str : num_strs) {
          if (ids.find(str) == ids.end()) {
            ids.insert(std::make_pair(str, vertex_count++));
          }
          edge_pairs.push_back(ids[str]);
        }
      }
      file.close();
    }
    ids.clear();

    std::cout << "edge pairs size=" << edge_pairs.size() << std::endl;
    assert(edge_pairs.size() % 2 == 0);
    edge_count = edge_pairs.size() / 2;
    if (!directed) {
      edge_count *= 2;
    }

    std::vector<uintE> offsets(vertex_count + 1, 0);
    for (size_t i = 0; i < edge_pairs.size(); i += 2) {
      offsets[edge_pairs[i]]++;
      if (!directed) {
        offsets[edge_pairs[i + 1]]++;
      }
    }

    row_ptrs = new uintE[vertex_count + 1];
    cols = new uintV[edge_count];

    uintE prefix = 0;
    for (uintV i = 0; i <= vertex_count; ++i) {
      row_ptrs[i] = prefix;
      prefix += offsets[i];
      offsets[i] = row_ptrs[i];
    }

    for (size_t i = 0; i < edge_pairs.size(); i += 2) {
      cols[offsets[edge_pairs[i]]++] = edge_pairs[i + 1];
      if (!directed) {
        cols[offsets[edge_pairs[i + 1]]++] = edge_pairs[i];
      }
    }

    offsets.clear();
    edge_pairs.clear();

    SortCSRArray(vertex_count, row_ptrs, cols);
    std::cout << "finish building CSR" << std::endl;
  }

  static void ReadSNAPFile(std::string& filename, bool directed,
                           size_t& vertex_count, size_t& edge_count,
                           uintE*& row_ptrs, uintV*& cols) {
    vertex_count = 0;
    edge_count = 0;
    row_ptrs = NULL;
    cols = NULL;

    std::cout << "start build csr..." << std::endl;
    uintV min_vertex_id = kMaxuintV;
    uintV max_vertex_id = kMinuintV;
    {
      std::ifstream file(filename.c_str(), std::fstream::in);
      std::string line;
      uintV vids[2];
      while (getline(file, line)) {
        if (line.length() == 0 || !std::isdigit(line[0])) continue;
        std::istringstream iss(line);
        for (int i = 0; i < 2; ++i) {
          iss >> vids[i];
          min_vertex_id = std::min(min_vertex_id, vids[i]);
          max_vertex_id = std::max(max_vertex_id, vids[i]);
        }
        edge_count++;
      }
      file.close();
    }
    vertex_count = max_vertex_id - min_vertex_id + 1;
    if (!directed) edge_count *= 2;
    std::cout << "vertex_count=" << vertex_count << ",edge_count=" << edge_count
              << std::endl;

    row_ptrs = new uintE[vertex_count + 1];
    cols = new uintV[edge_count];
    auto offsets = new uintE[vertex_count + 1];
    memset(offsets, 0, sizeof(uintE) * (vertex_count + 1));
    {
      std::ifstream file(filename.c_str(), std::fstream::in);
      std::string line;
      uintV vids[2];
      while (getline(file, line)) {
        if (line.length() == 0 || !std::isdigit(line[0])) continue;
        std::istringstream iss(line);
        for (int i = 0; i < 2; ++i) iss >> vids[i], vids[i] -= min_vertex_id;
        offsets[vids[0]]++;
        if (!directed) {
          offsets[vids[1]]++;
        }
      }
      file.close();
    }
    uintE prefix = 0;
    for (size_t i = 0; i < vertex_count + 1; ++i) {
      row_ptrs[i] = prefix;
      prefix += offsets[i];
      offsets[i] = row_ptrs[i];
    }

    {
      std::ifstream file(filename.c_str(), std::fstream::in);
      std::string line;
      uintV vids[2];
      while (getline(file, line)) {
        if (line.length() == 0 || !std::isdigit(line[0])) continue;
        std::istringstream iss(line);
        for (int i = 0; i < 2; ++i) iss >> vids[i], vids[i] -= min_vertex_id;
        cols[offsets[vids[0]]++] = vids[1];
        if (!directed) {
          cols[offsets[vids[1]]++] = vids[0];
        }
      }
      file.close();
    }
    delete[] offsets;
    offsets = NULL;

    SortCSRArray(vertex_count, row_ptrs, cols);
    std::cout << "finish building CSR" << std::endl;
  }

  // for the input of testing
  static void ReadFromVector(std::vector<std::vector<uintV>>& data,
                             size_t& vertex_count, size_t& edge_count,
                             uintE*& row_ptrs, uintV*& cols) {
    vertex_count = data.size();
    row_ptrs = new uintE[vertex_count + 1];
    uintE sum = 0;
    for (uintV u = 0; u < data.size(); ++u) {
      row_ptrs[u] = sum;
      sum += data[u].size();
    }
    row_ptrs[vertex_count] = sum;
    edge_count = sum;
    cols = new uintV[edge_count];

    for (uintV u = 0; u < data.size(); ++u) {
      for (uintE off = 0; off < data[u].size(); ++off) {
        cols[row_ptrs[u] + off] = data[u][off];
        assert(data[u][off] < vertex_count);
      }
    }
  }

  static void WriteDataFile(GraphFormat graph_format, std::string& filename,
                            bool directed, size_t vertex_count,
                            size_t edge_count, uintE* row_ptrs, uintV* cols) {
    switch (graph_format) {
      case SNAP:
        WriteCSRSNAPFile(filename, directed, vertex_count, edge_count, row_ptrs,
                         cols);
        break;
      case BIN:
        WriteCSRBinFile(filename, directed, vertex_count, edge_count, row_ptrs,
                        cols);
        break;
      case ADJ:
        WriteCSRAdjFile(filename, directed, vertex_count, edge_count, row_ptrs,
                        cols);
        break;
      default:
        assert(false);
    }
  }

  static void WriteCSRBinFile(std::string& filename, bool directed,
                              size_t vertex_count, size_t edge_count,
                              uintE* row_ptrs, uintV* cols) {
    TimeMeasurer timer;
    timer.StartTimer();
    std::cout << "start write csr bin file....";
    std::string output_filename = filename + ".bin";
    FILE* file_out = fopen(output_filename.c_str(), "wb");
    assert(file_out != NULL);
    size_t res = 0;
    size_t uintV_size = sizeof(uintV), uintE_size = sizeof(uintE);
    fwrite(&uintV_size, sizeof(size_t), 1, file_out);
    fwrite(&uintE_size, sizeof(size_t), 1, file_out);
    fwrite(&vertex_count, sizeof(size_t), 1, file_out);
    fwrite(&edge_count, sizeof(size_t), 1, file_out);
    for (uintV u = 0; u <= vertex_count; ++u) {
      res += fwrite(&row_ptrs[u], sizeof(uintE), 1, file_out);
    }
    for (uintV u = 0; u < vertex_count; ++u) {
      for (uintE j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        auto v = cols[j];
        res += fwrite(&v, sizeof(uintV), 1, file_out);
      }
    }
    assert(res == ((vertex_count + 1) + edge_count));
    fclose(file_out);
    timer.EndTimer();
    std::cout << "finish write csr bin file, output_filename="
              << output_filename
              << ",elapsed_time=" << timer.GetElapsedMicroSeconds() / 1000.0
              << "ms" << std::endl;
  }

  static void WriteCSRSNAPFile(std::string& filename, bool directed,
                               size_t vertex_count, size_t edge_count,
                               uintE* row_ptrs, uintV* cols) {
    TimeMeasurer timer;
    timer.StartTimer();
    std::cout << "start write csr snap file....";
    std::string output_filename = filename + ".snap.txt";
    std::ofstream file(output_filename.c_str(), std::fstream::out);
    for (uintV u = 0; u < vertex_count; ++u) {
      for (uintE j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        auto v = cols[j];
        if (u < v) {
          file << u << " " << v << std::endl;
        }
      }
    }
    file.close();
    timer.EndTimer();
    std::cout << "finish write csr snap file, output_filename="
              << output_filename
              << ",elapsed_time=" << timer.GetElapsedMicroSeconds() / 1000.0
              << "ms" << std::endl;
  }

  // for fennel
  static void WriteCSRAdjFile(std::string& filename, bool directed,
                              size_t vertex_count, size_t edge_count,
                              uintE* row_ptrs, uintV* cols) {
    TimeMeasurer timer;
    timer.StartTimer();
    std::cout << "start write csr adj file....";
    std::string output_filename = filename + ".adj.txt";
    std::ofstream file(output_filename.c_str(), std::fstream::out);
    for (uintV u = 0; u < vertex_count; ++u) {
      file << u << "\t";
      for (uintE j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        auto v = cols[j];
        file << v << " ";
      }
      file << std::endl;
    }
    file.close();

    std::string info_output_filename = filename + ".adj.txt.info";
    std::ofstream info_file(info_output_filename.c_str(), std::fstream::out);
    info_file << vertex_count << " " << edge_count;
    info_file.close();

    timer.EndTimer();
    std::cout << "finish write csr adj file, output_filename="
              << output_filename
              << ",elapsed_time=" << timer.GetElapsedMicroSeconds() / 1000.0
              << "ms" << std::endl;
  }

  static void SortCSRArray(size_t vertex_count, uintE* row_ptrs, uintV* cols) {
#if defined(OPENMP)
#pragma omp parallel for schedule(dynamic)
    for (uintV u = 0; u < vertex_count; ++u) {
      std::sort(cols + row_ptrs[u], cols + row_ptrs[u + 1]);
    }

#else
    std::cout << "start sorting..." << std::endl;
    for (uintV u = 0; u < vertex_count; ++u) {
      std::sort(cols + row_ptrs[u], cols + row_ptrs[u + 1]);
    }
#endif
  }

  static void Validate(size_t vertex_count, uintE* row_ptrs, uintV* cols) {
    // ensure edge list of each vertex is sorted
    // ensure no self loops
    // ensure no parallel edges
    size_t dangling = 0;
    size_t self_loops = 0;
    size_t parallel_edges = 0;
    for (uintV u = 0; u < vertex_count; ++u) {
      if (row_ptrs[u] == row_ptrs[u + 1]) {
        dangling++;
      }
      for (uintE j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) {
        auto v = cols[j];
        if (u == v) self_loops++;
        if (j > row_ptrs[u]) {
          if (v == cols[j - 1]) parallel_edges++;
        }
      }
    }
    if (dangling)
      std::cout << "Warning: dangling_nodes=" << dangling << std::endl;
    if (self_loops)
      std::cout << "Warning: self_loops=" << self_loops << std::endl;
    if (parallel_edges)
      std::cout << "Warning: parallel_edges=" << parallel_edges << std::endl;
    assert(dangling == 0 && self_loops == 0 && parallel_edges == 0);
  }
};

#endif
