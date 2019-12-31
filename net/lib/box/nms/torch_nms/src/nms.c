#include <TH/TH.h>
#include <math.h>

int cpu_nms(THLongTensor * keep_out, THLongTensor * num_out, THFloatTensor * boxes, THLongTensor * order, THFloatTensor * areas, float nms_overlap_thresh) {
    // boxes has to be sorted
    THArgCheck(THLongTensor_isContiguous(keep_out), 0, "keep_out must be contiguous");
    THArgCheck(THLongTensor_isContiguous(boxes), 2, "boxes must be contiguous");
    THArgCheck(THLongTensor_isContiguous(order), 3, "order must be contiguous");
    THArgCheck(THLongTensor_isContiguous(areas), 4, "areas must be contiguous");
    // Number of ROIs
    long boxes_num = THFloatTensor_size(boxes, 0);
    long boxes_dim = THFloatTensor_size(boxes, 1);

    long * keep_out_flat = THLongTensor_data(keep_out);
    float * boxes_flat = THFloatTensor_data(boxes);
    long * order_flat = THLongTensor_data(order);
    float * areas_flat = THFloatTensor_data(areas);

    THByteTensor* suppressed = THByteTensor_newWithSize1d(boxes_num);
    THByteTensor_fill(suppressed, 0);
    unsigned char * suppressed_flat =  THByteTensor_data(suppressed);

    // nominal indices
    int i, j;
    // sorted indices
    int _i, _j;
    // temp variables for box i's (the box currently under consideration)
    float ix0, iy0, iz0, id, ih, iw, iarea;
    float ix1, iy1, iz1;
    // variables for computing overlap with box j (lower scoring box)
    float xx0, yy0, zz0, dd, hh, ww;
    float xx1, yy1, zz1;
    float w, h, d;
    float inter, ovr;

    long num_to_keep = 0;
    for (_i=0; _i < boxes_num; ++_i) {
        i = order_flat[_i];
        if (suppressed_flat[i] == 1) {
            continue;
        }
        keep_out_flat[num_to_keep++] = i;
        iz0 = boxes_flat[i * boxes_dim + 1];
        iy0 = boxes_flat[i * boxes_dim + 2];
        ix0 = boxes_flat[i * boxes_dim + 3];
        id = boxes_flat[i * boxes_dim + 4];
        ih = boxes_flat[i * boxes_dim + 5];
        iw = boxes_flat[i * boxes_dim + 6];
        iz0 -= id / 2; iz1 = iz0 + id;
        iy0 -= ih / 2; iy1 = iy0 + ih;
        ix0 -= iw / 2; ix1 = ix0 + iw;
        iarea = areas_flat[i];
        for (_j = _i + 1; _j < boxes_num; ++_j) {
            j = order_flat[_j];
            if (suppressed_flat[j] == 1) {
                continue;
            }
            zz0 = boxes_flat[j * boxes_dim + 1];
            yy0 = boxes_flat[j * boxes_dim + 2];
            xx0 = boxes_flat[j * boxes_dim + 3];
            dd = boxes_flat[j * boxes_dim + 4];
            hh = boxes_flat[j * boxes_dim + 5];
            ww = boxes_flat[j * boxes_dim + 6];
            zz0 -= dd / 2; zz1 = zz0 + dd;
            yy0 -= hh / 2; yy1 = yy0 + hh;
            xx0 -= ww / 2; xx1 = xx0 + ww;

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

            // printf("iz0 %f, iy0 %f, ix0 %f, d %f\n", iz0, iy0, ix0, dd);
            // printf("iz1 %f, iy1 %f, ix1 %f\n", iz1, iy1, ix1);

            // printf("zz0 %f, yy0 %f, xx0 %f, id %f\n", zz0, yy0, xx0, id);
            // printf("zz1 %f, yy1 %f, xx1 %f\n", zz1, yy1, xx1);
            // printf("w %f, h %f, d %f\n", w, h, d);
            // printf("iarea: %f, ibarea: %f, inter: %f\n", iarea, areas_flat[j], inter);
            ovr = inter / (iarea + areas_flat[j] - inter);
            if (ovr >= nms_overlap_thresh) {
                suppressed_flat[j] = 1;
            }
        }
    }

    long *num_out_flat = THLongTensor_data(num_out);
    *num_out_flat = num_to_keep;
    THByteTensor_free(suppressed);
    return 1;
}