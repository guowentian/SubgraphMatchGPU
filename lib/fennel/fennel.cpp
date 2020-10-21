//
// Created by gongsf on 15/5/19.
//

#include <float.h>
#include <boost/algorithm/string/trim.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace boost;

long unfind(vector<long>& un_father, long u) {
  return un_father[u] == u ? u : (un_father[u] = unfind(un_father, un_father[u]));
}
void ununion(vector<long>& un_father, long u, long v) {
  un_father[u] = v;
}

void fennel(int argc, char *argv[]) {
  cout << "fennel start !!!!!!!!!!!" << endl;
  string input;
  string output;
  int part_num = -1;
  long vertex_num = -1;
  long edge_num = -1;

  for (int i = 0; i < argc; i++) {
    string arg = argv[i];
    if (arg == "--in") {
      input = argv[++i];
    } else if (arg == "--out") {
      output = argv[++i];
    } else if (arg == "--part_num") {
      part_num = atoi(argv[++i]);
    } else if (arg == "--vertex_num") {
      vertex_num = atol(argv[++i]);
    } else if (arg == "--edge_num") {
      edge_num = atol(argv[++i]);
    }
  }

  if (part_num == -1 || vertex_num == -1 || edge_num == -1) {
    cout << "invalie parameters" << endl;
    cout << part_num << endl;
    cout << vertex_num << endl;
    cout << edge_num << endl;
    exit(0);
  }

  ifstream infile(input.c_str());
  if (infile.is_open()) {
    string line;
    edge_num = edge_num / 2;
    float gamma = 1.5;
    float alpha = sqrt(part_num) * edge_num / pow(vertex_num, gamma);
    float v = 1.1;
    float miu = v * vertex_num / part_num;
    vector<long> part_info = vector<long>(vertex_num, 0);
    vector<long> vertex_num_part = vector<long>(part_num, 0);

    vector<long> un_father(vertex_num);
    vector<long> un_first(part_num, -1);
    for (long i = 0; i < vertex_num; ++i) un_father[i] = i;

    // the format each line is: u\tv1 v2 v3 ....
    for (long i = 0; getline(infile, line); i++) {
      if (i % 1000 == 0) cout << i << endl;
      trim(line);
      long pos = line.find_first_of("\t");
      long id = atoi(line.substr(0, pos).c_str());
      line = line.substr(pos + 1);
      vector<long> neighbor;
      long to = 0;
      for (size_t p = 0; p < line.length(); ++p) {
        if (line[p] >= '0' && line[p] <= '9') {
          to = to*10+(line[p] - '0');
        }
        if (p == line.length() -1 || line[p] == ' ') {
          neighbor.push_back(to);
          to = 0;
        }
      }

      float max_score = -FLT_MAX;
      int max_part = 0;
      for (long i = 0; i < part_num; i++) {
        if (vertex_num_part[i] <= miu) {
          double delta_c = alpha * (pow(vertex_num_part[i] + 1, gamma) -
                                    pow(vertex_num_part[i], gamma));
          float score = 0;
          for (long j = 0; j < neighbor.size(); j++) {
            long nid = neighbor[j];
            // this partition contains nid
            if (un_first[i] >= 0 && unfind(un_father, un_first[i]) == unfind(un_father, nid)) {
              score += 1;
            }
          }
          score = score - delta_c;
          if (max_score < score) {
            max_score = score;
            max_part = i;
          }
        }
      }
      if (un_first[max_part] < 0) un_first[max_part] = id;
      ununion(un_father, id, un_first[max_part]);
      vertex_num_part[max_part] += 1;
      part_info[id] = max_part;
    }
    ofstream outfile(output.c_str());
    for (long i = 0; i < vertex_num; i++) {
      outfile << part_info[i] << endl;
    }
    outfile.close();
  }
  infile.close();
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    std::cout << "./fennel --in [] --out [] --part_num [] --vertex_num [] "
                 "--edge_num []"
              << std::endl;
    return -1;
  }
  fennel(argc, argv);
  return 0;
}

