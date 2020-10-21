#ifndef __MODERN_GPU_CONTEXT_CUH__
#define __MODERN_GPU_CONTEXT_CUH__

#include <moderngpu/launch_box.hxx>

// typedef launch_box_t<arch_70_cta<128, 11, 9>, arch_61_cta<128, 11, 9>>
//    MGPULaunchBox;

typedef mgpu::launch_box_t<mgpu::arch_70_cta<128, 8, 8>,
                           mgpu::arch_61_cta<128, 8, 8>>
    MGPULaunchBox;

typedef mgpu::launch_box_t<mgpu::arch_70_cta<128, 1>, mgpu::arch_61_cta<128, 1>>
    MGPULaunchBoxVT1;

#endif
