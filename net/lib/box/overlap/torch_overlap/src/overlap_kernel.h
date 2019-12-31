#ifndef _NMS_KERNEL
#define _NMS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

void _overlap(int boxes1_num, float* boxes1_flat, int boxes2_num, float* boxes2_flat, float* overlap_flat);

#ifdef __cplusplus
}
#endif

#endif

