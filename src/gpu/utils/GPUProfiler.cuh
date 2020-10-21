#ifndef __GPU_UTILS_GPU_PROFILER_CUH__
#define __GPU_UTILS_GPU_PROFILER_CUH__

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "GPUTimer.cuh"
#include "GPUUtil.cuh"

class GPUProfiler {
 public:
  GPUProfiler(size_t stream_num) : stream_num_(stream_num) {
    phase_num_ = 0;
    timers_.resize(stream_num_);
    accum_times_.resize(stream_num_);
  }
  ~GPUProfiler() {
    for (size_t i = 0; i < stream_num_; ++i) {
      for (size_t j = 0; j < phase_num_; ++j) {
        delete timers_[i][j];
        timers_[i][j] = NULL;
      }
    }
  }
  void Clear() {
    for (size_t i = 0; i < stream_num_; ++i) {
      for (size_t j = 0; j < phase_num_; ++j) {
        accum_times_[i][j] = 0;
      }
    }
  }
  void AddPhase(const char *phase_name) {
#if defined(PROFILE)
    assert(!PhaseExists(phase_name));
    phase_names_.push_back(std::string(phase_name));
    phase_num_++;
    for (size_t i = 0; i < stream_num_; ++i) {
      CUDA_ERROR(cudaSetDevice(i));
      timers_[i].push_back(new GPUTimer());
      accum_times_[i].push_back(0.0);
    }
#endif
  }
  void StartTimer(const char *phase_name, size_t stream_id,
                  cudaStream_t stream) {
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    timers_[stream_id][pos]->StartTimer(stream);
#endif
  }
  void EndTimer(const char *phase_name, size_t stream_id, cudaStream_t stream) {
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    timers_[stream_id][pos]->EndTimer(stream);
    accum_times_[stream_id][pos] +=
        timers_[stream_id][pos]->GetElapsedMilliSeconds();
#endif
  }
  // in milliseconds
  float AggregatePhase(const char *phase_name) {
    double ret = 0.0;
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    for (size_t stream_id = 0; stream_id < stream_num_; ++stream_id) {
      ret += accum_times_[stream_id][pos];
    }
#endif
    return ret;
  }

  void Report(size_t stream_id) {
#if defined(PROFILE)
    std::cout << "stream_id=" << stream_id << ",";
    for (size_t j = 0; j < phase_num_; ++j) {
      if (j > 0) std::cout << ",";
      std::cout << phase_names_[j] << "=" << accum_times_[stream_id][j] << "ms";
    }
    std::cout << std::endl;
#endif
  }
  void Report() {
#if defined(PROFILE)
    for (size_t i = 0; i < stream_num_; ++i) {
      Report(i);
    }
#endif
  }
  void Report(const char *phase_name) {
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    std::cout << "phase_name=" << phase_name;
    for (size_t stream_id = 0; stream_id < stream_num_; ++stream_id) {
      std::cout << ",(stream_id=" << stream_id << ","
                << accum_times_[stream_id][pos] << "ms)";
    }
    std::cout << std::endl;
#endif
  }

  size_t GetStreamNum() const { return stream_num_; }
  size_t GetPhaseNum() const { return phase_num_; }

 private:
  size_t FindPhaseIndex(const char *phase_name) {
    std::string name(phase_name);
    size_t pos = phase_num_;
    for (size_t i = 0; i < phase_num_; ++i) {
      if (name == phase_names_[i]) {
        pos = i;
        break;
      }
    }
    assert(pos < phase_num_);
    return pos;
  }
  bool PhaseExists(const char *phase_name) {
    std::string name(phase_name);
    for (size_t i = 0; i < phase_names_.size(); ++i) {
      if (name == phase_names_[i]) {
        return true;
      }
    }
    return false;
  }

  size_t stream_num_;
  size_t phase_num_;
  std::vector<std::string> phase_names_;
  std::vector<std::vector<GPUTimer *> > timers_;
  std::vector<std::vector<float> > accum_times_;
};

#endif
