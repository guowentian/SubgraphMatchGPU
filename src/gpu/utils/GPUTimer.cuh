#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

#include <string>
#include <vector>
#include "GPUUtil.cuh"

class GPUTimer {
 public:
  GPUTimer() {
    CUDA_ERROR(cudaEventCreate(&start));
    CUDA_ERROR(cudaEventCreate(&stop));
  }
  ~GPUTimer() {
    CUDA_ERROR(cudaEventDestroy(start));
    CUDA_ERROR(cudaEventDestroy(stop));
  }
  void StartTimer(cudaStream_t stream = 0) {
    CUDA_ERROR(cudaEventRecord(start, stream));
  }
  void EndTimer(cudaStream_t stream = 0) {
    CUDA_ERROR(cudaEventRecord(stop, stream));
  }
  float GetElapsedMilliSeconds() {
    float ret;
    CUDA_ERROR(cudaEventSynchronize(stop));
    CUDA_ERROR(cudaEventElapsedTime(&ret, start, stop));
    return ret;
  }

 private:
  cudaEvent_t start;
  cudaEvent_t stop;
};

#endif
