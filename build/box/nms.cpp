#include <torch/extension.h>
#include <iostream>
#include <math.h>

int cpu_nms(at::Tensor * keep_out, at::Tensor * num_out, at::Tensor * boxes, at::Tensor * order, at::Tensor * areas, float nms_overlap_thresh) {
    // boxes has to be sorted
    AT_CHECK(keep_out->is_contiguous(), "argument#1(keep_out) must be contiguous");
    AT_CHECK(order->is_contiguous(), "argument#4(order) must be contiguous");
    AT_CHECK(boxes->is_contiguous(), "argument#3(boxes) must be contiguous");
    AT_CHECK(areas->is_contiguous(), "argument#5(areas) must be contiguous");

    // Number of ROIs
    long boxes_num = boxes->sizes()[0];
    long boxes_dim = boxes->sizes()[1];

    auto keep_out_flat = keep_out->data<long>();
    auto boxes_flat = boxes->data<float>();
    auto order_flat = order->data<long>();
    auto areas_flat =  areas->data<float>();

    at::Tensor suppressed = at::zeros({boxes_num}, at::kInt);
    auto suppressed_flat = suppressed.data<int>();

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

    auto num_out_flat = num_out->data<long>();
    *num_out_flat = num_to_keep;

    return 1;
}
