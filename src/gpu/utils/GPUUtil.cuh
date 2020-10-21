#ifndef __GPU_UTIL_H__
#define __GPU_UTIL_H__

#include <cstdio>
#include <cub/cub.cuh>
#include <vector>
#include "Meta.h"

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file,
            line);
    exit(-1);
  }
}

#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define CALC_BLOCK_NUM(work, threads) (work + threads - 1) / threads

#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __forceinline__ __device__ __host__

__device__ static double atomicMul(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val * __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

// ================= device functions ===============

DEVICE void BlockSync() { __syncthreads(); }
DEVICE int BlockSyncOr(int pred) { return __syncthreads_or(pred); }
DEVICE void WarpSync() { cub::WARP_SYNC(FULL_WARP_MASK); }
DEVICE int WarpSyncOr(int pred) { return cub::WARP_ANY(pred, FULL_WARP_MASK); }
// return the mask indicating threads that have the same value,
// pred is set to true if all participating threads have the same values
template <typename T>
DEVICE unsigned int WarpMatchAll(T value, int *pred) {
  return __match_all_sync(FULL_WARP_MASK, value, pred);
}
template <typename T>
DEVICE unsigned int WarpMatchAll(unsigned int mask, T value, int *pred) {
  return __match_all_sync(mask, value, pred);
}

DEVICE unsigned int WarpBallot(int pred) {
  return __ballot_sync(FULL_WARP_MASK, pred);
}
DEVICE unsigned int WarpBallot(unsigned int mask, int pred) {
  return __ballot_sync(mask, pred);
}

template <typename T>
DEVICE T WarpShfl(T value, int src_lane) {
  return __shfl_sync(FULL_WARP_MASK, value, src_lane);
}

template <typename T>
HOST_DEVICE void Swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}
template <typename T>
HOST_DEVICE T Min(const T &a, const T &b) {
  return (a < b) ? a : b;
}
template <typename T>
HOST_DEVICE T Max(const T &a, const T &b) {
  return (a > b) ? a : b;
}

// ================== vector types ========================
typedef uint2 IndexTypeTuple2;

// ================== memory operations ========================
template <typename T>
void DToH(T *dest, const T *source, size_t count) {
  CUDA_ERROR(
      cudaMemcpy(dest, source, count * sizeof(T), cudaMemcpyDeviceToHost));
}
template <typename T>
void DToD(T *dest, const T *source, size_t count) {
  CUDA_ERROR(
      cudaMemcpy(dest, source, sizeof(T) * count, cudaMemcpyDeviceToDevice));
}
template <typename T>
void HToD(T *dest, const T *source, size_t count) {
  CUDA_ERROR(
      cudaMemcpy(dest, source, sizeof(T) * count, cudaMemcpyHostToDevice));
}
template <typename T>
void DToH(T *dest, const T *source, size_t count, cudaStream_t stream) {
  CUDA_ERROR(cudaMemcpyAsync(dest, source, count * sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
}
template <typename T>
void DToD(T *dest, const T *source, size_t count, cudaStream_t stream) {
  CUDA_ERROR(cudaMemcpyAsync(dest, source, sizeof(T) * count,
                             cudaMemcpyDeviceToDevice, stream));
}
template <typename T>
void HToD(T *dest, const T *source, size_t count, cudaStream_t stream) {
  CUDA_ERROR(cudaMemcpyAsync(dest, source, sizeof(T) * count,
                             cudaMemcpyHostToDevice, stream));
}

template <typename T>
void DToH(std::vector<T> &dest, const T *source, size_t count) {
  dest.resize(count);
  CUDA_ERROR(cudaMemcpy(dest.data(), source, sizeof(T) * count,
                        cudaMemcpyDeviceToHost));
}
template <typename T>
void HToD(T *dest, const std::vector<T> &source, size_t count) {
  CUDA_ERROR(cudaMemcpy(dest, source.data(), sizeof(T) * count,
                        cudaMemcpyHostToDevice));
}

template <typename T>
T GetD(T *source) {
  T ret;
  DToH(&ret, source, 1);
  return ret;
}
template <typename T>
T GetD(T *source, cudaStream_t stream) {
  T ret;
  DToH(&ret, source, 1, stream);
  return ret;
}
template <typename T>
void SetD(T *dest, T v) {
  HToD(dest, &v, 1);
}
template <typename T>
void SetD(T *dest, T v, cudaStream_t stream) {
  HToD(dest, &v, 1, stream);
}

#endif
