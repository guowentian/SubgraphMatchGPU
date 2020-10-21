#ifndef __UTILS_PHASE_PROFILER_H__
#define __UTILS_PHASE_PROFILER_H__

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "TimeMeasurer.h"

class PhaseProfiler {
 public:
  PhaseProfiler(size_t thread_num) : thread_num_(thread_num) {
    phase_num_ = 0;
    timers_.resize(thread_num_);
    accum_times_.resize(thread_num_);
  }
  ~PhaseProfiler() {
    for (size_t i = 0; i < thread_num_; ++i) {
      for (size_t j = 0; j < phase_num_; ++j) {
        delete timers_[i][j];
        timers_[i][j] = NULL;
      }
    }
  }
  void Clear() {
    for (size_t i = 0; i < thread_num_; ++i) {
      for (size_t j = 0; j < phase_num_; ++j) {
        accum_times_[i][j] = 0;
      }
    }
  }
  void AddPhase(const char *phase_name) {
    assert(!PhaseExists(phase_name));
    phase_names_.push_back(std::string(phase_name));
    phase_num_++;
    for (size_t i = 0; i < thread_num_; ++i) {
      timers_[i].push_back(new TimeMeasurer());
      accum_times_[i].push_back(0.0);
    }
  }
  void StartTimer(const char *phase_name, size_t thread_id) {
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    timers_[thread_id][pos]->StartTimer();
#endif
  }
  void EndTimer(const char *phase_name, size_t thread_id) {
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    timers_[thread_id][pos]->EndTimer();
    accum_times_[thread_id][pos] +=
        timers_[thread_id][pos]->GetElapsedMicroSeconds();
#endif
  }
  // in milliseconds
  double AggregatePhase(const char *phase_name) {
    double ret = 0.0;
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      ret += accum_times_[thread_id][pos];
    }
#endif
    return ret;
  }

  void Report(size_t thread_id) {
#if defined(PROFILE)
    if (phase_num_) {
      std::cout << "thread_id=" << thread_id << ",";
      for (size_t j = 0; j < phase_num_; ++j) {
        if (j > 0) std::cout << ",";
        std::cout << phase_names_[j] << "="
                  << accum_times_[thread_id][j] / 1000.0 << "ms";
      }
      std::cout << std::endl;
    }
#endif
  }
  void Report() {
#if defined(PROFILE)
    for (size_t i = 0; i < thread_num_; ++i) {
      Report(i);
    }
#endif
  }
  void Report(const char *phase_name) {
#if defined(PROFILE)
    size_t pos = FindPhaseIndex(phase_name);
    std::cout << "phase_name=" << phase_name;
    for (size_t thread_id = 0; thread_id < thread_num_; ++thread_id) {
      std::cout << ",(thread_id=" << thread_id << ","
                << accum_times_[thread_id][pos] << "ms)";
    }
    std::cout << std::endl;
#endif
  }

  size_t GetThreadNum() const { return thread_num_; }
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

  size_t thread_num_;
  size_t phase_num_;
  std::vector<std::string> phase_names_;
  std::vector<std::vector<TimeMeasurer *> > timers_;
  std::vector<std::vector<double> > accum_times_;
};
#endif
