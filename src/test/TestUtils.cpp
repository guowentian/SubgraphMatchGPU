#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "CPUParallelIntersection.h"
#include "ParallelUtils.h"

TEST(ParallelUtils, Scan) {
  const int N = 100;
  std::vector<int> input(N);
  std::vector<int> exp_output(N);
  int sum = 0;
  for (int i = 0; i < N; ++i) {
    exp_output[i] = sum;
    input[i] = i;
    sum += i;
  }
  auto func = [=](int left, int right) { return left + right; };

  std::vector<int> output(N);
  int actual_sum =
      ParallelUtils::ParallelScan(input.data(), output.data(), N, func);

  ASSERT_EQ(actual_sum, sum);
  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(output[i], exp_output[i]);
  }
}
TEST(ParallelUtils, Reduce) {
  const int N = 100;
  std::vector<int> input(N);
  std::vector<int> exp_output(N);
  int sum = 0;
  for (int i = 0; i < N; ++i) {
    exp_output[i] = sum;
    input[i] = i;
    sum += i;
  }
  auto func = [=](int left, int right) { return left + right; };

  std::vector<int> output(N);
  int actual_sum = ParallelUtils::ParallelReduce(input.data(), N, func);

  ASSERT_EQ(actual_sum, sum);
}
TEST(ParallelUtils, Pack) {
  const int N = 100;
  std::vector<int> input(N);
  std::vector<int> exp_output;
  bool* flag = new bool[N];
  for (int i = 0; i < N; ++i) {
    input[i] = i;
    flag[i] = i % 2;
    if (flag[i]) {
      exp_output.push_back(i);
    }
  }

  std::vector<int> output(N);
  size_t actual_num = ParallelUtils::Pack(input.data(), output.data(), flag, N);

  ASSERT_EQ(actual_num, exp_output.size());
  for (int i = 0; i < exp_output.size(); ++i) {
    ASSERT_EQ(output[i], exp_output[i]);
  }

  delete flag;
  flag = NULL;
}

TEST(CheckIntersect, ParallelIntersect) {
  const int segments_num = 1000;
  const int count_per_segment = 100;
  std::vector<int> input1_row_ptrs(segments_num + 1);
  std::vector<int> input1_inst(segments_num * count_per_segment);
  std::vector<int> input2_row_ptrs(segments_num + 1);
  std::vector<int> input2_inst(segments_num * count_per_segment);
  std::vector<int> exp_output_row_ptrs(segments_num + 1);
  std::vector<int> exp_output_inst;

  int prefix_sum = 0;
  int output_prefix_sum = 0;
  for (int i = 0; i < segments_num; ++i) {
    input1_row_ptrs[i] = prefix_sum;
    input2_row_ptrs[i] = prefix_sum;
    exp_output_row_ptrs[i] = output_prefix_sum;

    int value = 0;
    for (int pos = prefix_sum; pos < prefix_sum + count_per_segment; ++pos) {
      input1_inst[pos] = value;
      input2_inst[pos] = value + 2;
      ++value;
    }
    value = 0;
    for (int pos = prefix_sum; pos < prefix_sum + count_per_segment - 2;
         ++pos) {
      exp_output_inst.push_back(value + 2);
      ++value;
    }

    output_prefix_sum += count_per_segment - 2;
    prefix_sum += count_per_segment;
  }
  input1_row_ptrs[segments_num] = prefix_sum;
  input2_row_ptrs[segments_num] = prefix_sum;
  exp_output_row_ptrs[segments_num] = output_prefix_sum;

  std::vector<int*> edges1_start_ptrs(segments_num);
  std::vector<int*> edges1_end_ptrs(segments_num);
  std::vector<int*> edges2_start_ptrs(segments_num);
  std::vector<int*> edges2_end_ptrs(segments_num);
  for (int i = 0; i < segments_num; ++i) {
    edges1_start_ptrs[i] = input1_inst.data() + input1_row_ptrs[i];
    edges1_end_ptrs[i] = input1_inst.data() + input1_row_ptrs[i + 1];
    edges2_start_ptrs[i] = input2_inst.data() + input2_row_ptrs[i];
    edges2_end_ptrs[i] = input2_inst.data() + input2_row_ptrs[i + 1];
  }

  std::vector<int> output_row_ptrs;
  std::vector<int> output_inst;
  ParallelIntersect(edges1_start_ptrs, edges1_end_ptrs, edges2_start_ptrs,
                    edges2_end_ptrs, segments_num, output_row_ptrs,
                    output_inst);

  for (int i = 0; i < output_row_ptrs.size(); ++i) {
    ASSERT_EQ(output_row_ptrs[i], exp_output_row_ptrs[i]);
  }
  for (int i = 0; i < output_inst.size(); ++i) {
    ASSERT_EQ(output_inst[i], exp_output_inst[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ParallelUtils::SetParallelThreadNumber(10);

  return RUN_ALL_TESTS();
}
