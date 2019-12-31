// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdio.h>
#include <float.h>
#include "nms_kernel.h"

__device__ inline float devIoU(float const * const a, float const * const b) {
  /*
  float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  float width = fmaxf(right - left + 1, 0.f), height = fmaxf(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
  */

  float ix0, iy0, iz0, id;
  float ix1, iy1, iz1;
  float xx0, yy0, zz0, dd;
  float xx1, yy1, zz1;
  float w, h, d;
  float inter;

  iz0 = a[1];
  iy0 = a[2];
  ix0 = a[3];
  id = a[4];
  iz0 -= id / 2; iz1 = iz0 + id;
  iy0 -= id / 2; iy1 = iy0 + id;
  ix0 -= id / 2; ix1 = ix0 + id;

  zz0 = b[1];
  yy0 = b[2];
  xx0 = b[3];
  dd = b[4];
  zz0 -= dd / 2; zz1 = zz0 + dd;
  yy0 -= dd / 2; yy1 = yy0 + dd;
  xx0 -= dd / 2; xx1 = xx0 + dd;

  zz0 = fmaxf(iz0, zz0);
  yy0 = fmaxf(iy0, yy0);
  xx0 = fmaxf(ix0, xx0);

  zz1 = fminf(iz1, zz1);
  yy1 = fminf(iy1, yy1);
  xx1 = fminf(ix1, xx1);

  w = fmaxf(0.0, xx1 - xx0);
  h = fmaxf(0.0, yy1 - yy0);
  d = fmaxf(0.0, zz1 - zz0);
  inter = w * h * d;

  float Sa = a[4] * a[4] * a[4];
  float Sb = b[4] * b[4] * b[4];

  // printf("iz0 %f, iy0 %f, ix0 %f, d %f\n", iz0, iy0, ix0, dd);
  // printf("iz1 %f, iy1 %f, ix1 %f\n", iz1, iy1, ix1);

  // printf("zz0 %f, yy0 %f, xx0 %f, id %f\n", zz0, yy0, xx0, id);
  // printf("zz1 %f, yy1 %f, xx1 %f\n", zz1, yy1, xx1);
  // printf("w %f, h %f, d %f\n", w, h, d);
  // printf("iarea: %f, ibarea: %f, inter: %f\n", Sa, Sb, inter);
  return inter / (Sa + Sb - inter);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


void _nms(int boxes_num, float * boxes_dev,
          unsigned long long * mask_dev, float nms_overlap_thresh) {

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);
}

#ifdef __cplusplus
}
#endif
