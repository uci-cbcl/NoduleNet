// int cpu_overlap(THLongTensor * keep_out, THLongTensor * num_out, THFloatTensor * boxes, THLongTensor * order, THFloatTensor * areas, float nms_overlap_thresh);

int cpu_overlap(THFloatTensor * boxes1, THFloatTensor * boxes2, THFloatTensor * overlap);