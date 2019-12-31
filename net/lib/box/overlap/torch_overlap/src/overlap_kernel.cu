// ------------------------------------------------------------------
// Written by Hao Tang
// ------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdio.h>
#include <float.h>
#include "overlap_kernel.h"

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

  iz0 = a[0];
  iy0 = a[1];
  ix0 = a[2];
  id = a[3];
  iz0 -= id / 2; iz1 = iz0 + id;
  iy0 -= id / 2; iy1 = iy0 + id;
  ix0 -= id / 2; ix1 = ix0 + id;

  zz0 = b[0];
  yy0 = b[1];
  xx0 = b[2];
  dd = b[3];
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

  float Sa = a[3] * a[3] * a[3];
  float Sb = b[3] * b[3] * b[3];

  // printf("iz0 %f, iy0 %f, ix0 %f, d %f\n", iz0, iy0, ix0, dd);
  // printf("iz1 %f, iy1 %f, ix1 %f\n", iz1, iy1, ix1);

  // printf("zz0 %f, yy0 %f, xx0 %f, id %f\n", zz0, yy0, xx0, id);
  // printf("zz1 %f, yy1 %f, xx1 %f\n", zz1, yy1, xx1);
  // printf("w %f, h %f, d %f\n", w, h, d);
  // printf("iarea: %f, ibarea: %f, inter: %f\n", Sa, Sb, inter);
  return inter / (Sa + Sb - inter);
}

__global__ void nms_kernel(const int n_boxes1, const float* boxes1, const int n_boxes2, const float* boxes2, float* overlap) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;
  // printf("blockIdx.y %d, blockIdx.x %d, thredIdx.x %d\n", blockIdx.y, blockIdx.x, threadIdx.x);

  const int row_size =
        fminf(n_boxes1 - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        fminf(n_boxes2 - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        boxes2[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        boxes2[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        boxes2[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        boxes2[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
    // printf("boxes2 [%f, %f, %f, %f]\n", boxes2[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0],
      // boxes2[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1], boxes2[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2],
      // boxes2[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3]);
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int box1_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *box1 = boxes1 + box1_idx * 4;
    int i = 0;
    int start = 0;
    // if (row_start == col_start) {
    //   start = threadIdx.x + 1;
    // }
    for (i = start; i < col_size; i++) {
      const int box2_idx = threadsPerBlock * col_start + i;
      const int index = box1_idx * n_boxes2 + box2_idx;
      // printf("box1 index %d, box2 index %d\n", box1_idx, box2_idx);
      overlap[index] = devIoU(box1, block_boxes + i * 4);
    }
  }
}


void _overlap(int boxes1_num, float* boxes1_flat, int boxes2_num, float* boxes2_flat, float* overlap_flat) {

  dim3 blocks(DIVUP(boxes2_num, threadsPerBlock),
              DIVUP(boxes1_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes1_num, boxes1_flat, boxes2_num, boxes2_flat, overlap_flat);
}

#ifdef __cplusplus
}
#endif
