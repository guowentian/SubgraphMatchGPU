#ifndef __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_VERIFY_COUNT_CUH__
#define __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_VERIFY_COUNT_CUH__

#include "LIGHTCount.cuh"

#include "Transform.cuh"

namespace Light {

static void VerifyConsecutiveSequence(uintV* d_array, size_t path_num,
                                      CudaContext* context) {
  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        if (d_array[index] != index) {
          printf("%d,%d\n", d_array[index], index);
        }
        assert(d_array[index] == index);
      },
      path_num, context);
}

static void VerifyFiliterCandidate(
    size_t d_partition_id, CudaContext* context, uintV filter_level,
    DeviceArray<size_t>* compact_path_ids, size_t path_num,
    DeviceArray<size_t>* new_candidates_offsets,
    DeviceArray<uintV>* new_candidates, ImData* im_data,
    ImDataDevHolder* im_data_holder, DevCondArrayType* cond,
    GPUProfiler* gpu_profiler, CountProfiler* count_profiler) {
  size_t* h_compact_path_ids = new size_t[path_num];
  DToH(h_compact_path_ids, compact_path_ids->GetArray(), path_num);
  assert(path_num == compact_path_ids->GetSize());

  DevCondArrayType h_cond;
  DToH(&h_cond, cond, 1);
  h_cond.Print();

  auto& d_instances = im_data->GetInstances();
  size_t full_path_num = d_instances[0]->GetSize();
  auto& d_candidates = im_data->GetCandidates();
  auto& d_candidates_indices = im_data->GetCandidatesIndices();
  auto& d_candidates_offsets = im_data->GetCandidatesOffsets();

  std::vector<uintV*> h_instances(kMaxQueryVerticesNum, NULL);
  for (size_t i = 0; i < h_cond.GetCount(); ++i) {
    uintV u = h_cond.Get(i).GetOperand();
    h_instances[u] = new uintV[full_path_num];
    assert(full_path_num == d_instances[u]->GetSize());
    DToH(h_instances[u], d_instances[u]->GetArray(), full_path_num);
  }

  size_t* h_cur_candidates_indices = new size_t[full_path_num];
  DToH(h_cur_candidates_indices, d_candidates_indices[filter_level]->GetArray(),
       full_path_num);
  size_t* h_cur_candidates_offsets =
      new size_t[d_candidates_offsets[filter_level]->GetSize()];
  DToH(h_cur_candidates_offsets, d_candidates_offsets[filter_level]->GetArray(),
       d_candidates_offsets[filter_level]->GetSize());
  uintV* h_cur_candidates = new uintV[d_candidates[filter_level]->GetSize()];
  DToH(h_cur_candidates, d_candidates[filter_level]->GetArray(),
       d_candidates[filter_level]->GetSize());

  std::vector<size_t> h_new_candidates_offsets;
  std::vector<uintV> h_new_candidates;

  size_t candidates_prefix_count = 0;
  for (size_t i = 0; i < path_num; ++i) {
    size_t path_id = h_compact_path_ids[i];
    size_t pos = h_cur_candidates_indices[path_id];
    uintV* candidates = h_cur_candidates + h_cur_candidates_offsets[pos];
    size_t candidates_count =
        h_cur_candidates_offsets[pos + 1] - h_cur_candidates_offsets[pos];

    uintV M[kMaxQueryVerticesNum] = {kMaxuintV};
    for (size_t cond_id = 0; cond_id < h_cond.GetCount(); ++cond_id) {
      uintV u = h_cond.Get(cond_id).GetOperand();
      M[u] = h_instances[u][path_id];
    }

    h_new_candidates_offsets.push_back(candidates_prefix_count);
    size_t segment_valid_count = 0;

    for (size_t j = 0; j < candidates_count; ++j) {
      uintV candidate = candidates[j];
      bool valid = true;
      for (size_t cond_id = 0; cond_id < h_cond.GetCount(); ++cond_id) {
        uintV u = h_cond.Get(cond_id).GetOperand();
        auto op = h_cond.Get(cond_id).GetOperator();
        if (op == LESS_THAN) {
          if (!(candidate < M[u])) valid = false;
        } else if (op == LARGER_THAN) {
          if (!(candidate > M[u])) valid = false;
        } else if (op == NON_EQUAL) {
          if (!(candidate != M[u])) valid = false;
        }
      }

      if (valid) {
        h_new_candidates.push_back(candidate);
        segment_valid_count++;
      }
    }

    candidates_prefix_count += segment_valid_count;
  }
  h_new_candidates_offsets.push_back(candidates_prefix_count);

  if (new_candidates->GetSize() == 0) {
    assert(h_new_candidates_offsets[path_num] == 0);
    assert(h_new_candidates.size() == 0);
  } else {
    size_t* h_copy_new_candidates_offsets =
        new size_t[new_candidates_offsets->GetSize()];
    uintV* h_copy_new_candidates = new uintV[new_candidates->GetSize()];
    DToH(h_copy_new_candidates_offsets, new_candidates_offsets->GetArray(),
         new_candidates_offsets->GetSize());
    DToH(h_copy_new_candidates, new_candidates->GetArray(),
         new_candidates->GetSize());

    assert(h_new_candidates_offsets.size() ==
           new_candidates_offsets->GetSize());
    assert(h_new_candidates.size() == new_candidates->GetSize());

    for (size_t i = 0; i < h_new_candidates_offsets.size(); ++i) {
      assert(h_new_candidates_offsets[i] == h_copy_new_candidates_offsets[i]);
    }
    for (size_t i = 0; i < h_new_candidates.size(); ++i) {
      assert(h_new_candidates[i] == h_copy_new_candidates[i]);
    }

    delete[] h_copy_new_candidates;
    h_copy_new_candidates = NULL;
    delete[] h_copy_new_candidates_offsets;
    h_copy_new_candidates_offsets = NULL;
  }

  delete[] h_cur_candidates_indices;
  h_cur_candidates_indices = NULL;
  delete[] h_cur_candidates_offsets;
  h_cur_candidates_offsets = NULL;
  delete[] h_cur_candidates;
  h_cur_candidates = NULL;

  for (size_t i = 0; i < h_cond.GetCount(); ++i) {
    uintV u = h_cond.Get(i).GetOperand();
    delete[] h_instances[u];
    h_instances[u] = NULL;
  }
  delete[] h_compact_path_ids;
  h_compact_path_ids = NULL;
}

}  // namespace Light

#endif