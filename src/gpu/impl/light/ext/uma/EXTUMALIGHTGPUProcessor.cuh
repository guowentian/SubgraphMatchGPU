#ifndef __EXTERNAL_UMA_GPU_LIGHT_PIPELINE_PROCESSOR_CUH__
#define __EXTERNAL_UMA_GPU_LIGHT_PIPELINE_PROCESSOR_CUH__

#include "EXTLIGHTGPUProcessor.cuh"

namespace Light {
class EXTUMALIGHTGPUProcessor : public EXTLIGHTGPUProcessor {
 public:
  EXTUMALIGHTGPUProcessor(LightWorkContext* wctx, Task* task,
                          bool incremental_load_subgraph)
      : EXTLIGHTGPUProcessor(wctx, task, incremental_load_subgraph) {}

  virtual void LoadSubgraph(LightWorkContext* wctx, DevConnType* conn,
                            size_t path_num) {
    // no need to load the subgraph into GPU memory as now we load the graph by
    // UMA, which loads the graph data on demand by hardware scheduling
  }
};
}  // namespace Light

#endif