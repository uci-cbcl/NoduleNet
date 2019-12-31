// ------------------------------------------------------------------
// Written by Hao Tang
// ------------------------------------------------------------------
#include <THC/THC.h>
#include <TH/TH.h>
#include <math.h>
#include <stdio.h>

#include "overlap_kernel.h"

extern THCState *state;

int gpu_overlap(THFloatTensor * boxes1, THFloatTensor * boxes2, THFloatTensor * overlap) {
  // boxes has to be sorted
  THArgCheck(THCudaTensor_isContiguous(state, boxes1), 0, "boxes1 must be contiguous");
  THArgCheck(THCudaTensor_isContiguous(state, boxes2), 1, "boxes2 must be contiguous");
  THArgCheck(THCudaTensor_isContiguous(state, overlap), 2, "overlap must be contiguous");
  // Number of ROIs
  int boxes1_num = THCudaTensor_size(state, boxes1, 0);
  int boxes1_dim = THCudaTensor_size(state, boxes1, 1);
  int boxes2_num = THCudaTensor_size(state, boxes2, 0);
  int boxes2_dim = THCudaTensor_size(state, boxes2, 1);

  float* boxes1_flat = THCudaTensor_data(state, boxes1);
  float* boxes2_flat = THCudaTensor_data(state, boxes2);
  float* overlap_flat = THCudaTensor_data(state, overlap);

  _overlap(boxes1_num, boxes1_flat, boxes2_num, boxes2_flat, overlap_flat);

  return 1;
}
