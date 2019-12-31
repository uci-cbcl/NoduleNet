from common import *
from net.lib.box.overlap.cython_box_overlap import cython_box_overlap
from net.lib.box.nms.torch_nms import torch_nms
from net.lib.box.nms.gpu_nms import gpu_nms
from net.lib.box.nms.cython_nms import cython_nms

#from model.lib.box.nms.py_nms import py_nms
#


## torch #####################################################################

# Clip process to image boundaries.
def torch_clip_boxes(boxes, width, height):
    boxes = torch.stack(
        (boxes[:,0].clamp(0, width  - 1),
         boxes[:,1].clamp(0, height - 1),
         boxes[:,2].clamp(0, width  - 1),
         boxes[:,3].clamp(0, height - 1)), 1)

    return boxes


def torch_box_transform(boxes, targets):
  bw = boxes[:, 2] - boxes[:, 0] + 1.0
  bh = boxes[:, 3] - boxes[:, 1] + 1.0
  bx = boxes[:, 0] + 0.5 * bw
  by = boxes[:, 1] + 0.5 * bh

  tw = targets[:, 2] - targets[:, 0] + 1.0
  th = targets[:, 3] - targets[:, 1] + 1.0
  tx = targets[:, 0] + 0.5 * tw
  ty = targets[:, 1] + 0.5 * th

  dx = (tx - bx) / bw
  dy = (ty - by) / bh
  dw = torch.log(tw / bw)
  dh = torch.log(th / bh)

  deltas = torch.stack((dx, dy, dw, dh), 1)
  return deltas


def torch_box_transform_inv(boxes, deltas):

    bw = boxes[:, 2] - boxes[:, 0] + 1.0
    bh = boxes[:, 3] - boxes[:, 1] + 1.0
    bx = boxes[:, 0] + 0.5 * bw
    by = boxes[:, 1] + 0.5 * bh
    bw = bw.unsqueeze(1)
    bh = bh.unsqueeze(1)
    bx = bx.unsqueeze(1)
    by = by.unsqueeze(1)

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    x = bx + dx * bw
    y = by + dy * bh
    w = torch.exp(dw) * bw
    h = torch.exp(dh) * bh

    predictions = torch.cat( (
        x - 0.5 * w,
        y - 0.5 * h,
        x + 0.5 * w,
        y + 0.5 * h), 1)

    return predictions


def torch_filter_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = (((ws >= min_size) + (hs >= min_size))==2).nonzero().view(-1)
    return keep


def torch_box_overlap(boxes, gt_boxes):

    '''
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and gt_boxes
    '''

    box_areas = (boxes   [:, 2] - boxes   [:, 0] + 1) *  (boxes   [:, 3] - boxes   [:, 1] + 1)
    gt_areas  = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *  (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)

    intersect_ws = (torch.min(boxes[:, 2:3], gt_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], gt_boxes[:, 0:1].t()) + 1).clamp(min=0)
    intersect_hs = (torch.min(boxes[:, 3:4], gt_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], gt_boxes[:, 1:2].t()) + 1).clamp(min=0)
    intersect_areas = intersect_ws * intersect_hs
    union_areas     = box_areas.view(-1, 1) + gt_areas.view(1, -1) - intersect_areas
    overlaps        = intersect_areas / union_areas

    return overlaps


# def torch_clip_boxes(boxes, width, height):
#     ''' Clip process to image boundaries. '''
#
#     boxes = torch.stack(
#         (boxes[:,0].clamp(0, width  - 1),
#          boxes[:,1].clamp(0, height - 1),
#          boxes[:,2].clamp(0, width  - 1),
#          boxes[:,3].clamp(0, height - 1)), 1)
#
#     return boxes





## python  #####################################################################

# # overlaps
# #https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
# #http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
#
# def one_box_overlap(box1,box2):
#
#     ''' Calculate the Intersection over Union (IoU) of two bounding boxes.
#
#     '''
#     #box=x0,y0,x1,y1
#     assert box1[0] < box1[2]
#     assert box1[1] < box1[3]
#     assert box2[0] < box2[2]
#     assert box2[1] < box2[3]
#
#     # determine the coordinates of the intersection rectangle
#     x_left   = max(box1[0], box2[0])
#     y_top    = max(box1[1], box2[1])
#     x_right  = min(box1[2], box2[2])
#     y_bottom = min(box1[3], box2[3])
#
#     if x_right < x_left or y_bottom < y_top:
#         return 0.0
#
#     # The intersection of two axis-aligned bounding boxes is always an
#     # axis-aligned bounding box
#     intersection_area = (x_right - x_left+1) * (y_bottom - y_top+1)
#
#     # compute the area of both AABBs
#     area1 = (box1[2] - box1[0]+1) * (box1[3] - box1[1]+1)
#     area2 = (box2[2] - box2[0]+1) * (box2[3] - box2[1]+1)
#
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = intersection_area / float(area1 + area2 - intersection_area)
#     assert iou >= 0.0
#     assert iou <= 1.0
#     return iou
#
#


def box_overlap(boxes, gt_boxes):

    box_areas = (boxes   [:, 2] - boxes   [:, 0] + 1) *  (boxes   [:, 3] - boxes   [:, 1] + 1)
    gt_areas  = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *  (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)

    intersect_ws = np.clip((np.min(boxes[:, 2:3], gt_boxes[:, 2:3].t()) - np.max(boxes[:, 0:1], gt_boxes[:, 0:1].t()) + 1),0,1e8)
    intersect_hs = np.clip((np.min(boxes[:, 3:4], gt_boxes[:, 3:4].t()) - np.max(boxes[:, 1:2], gt_boxes[:, 1:2].t()) + 1),0,1e8)
    intersect_areas = intersect_ws * intersect_hs
    union_areas     = box_areas.view(-1, 1) + gt_areas.view(1, -1) - intersect_areas
    overlaps        = intersect_areas / union_areas

    return overlaps





## python  ##############################################################################
def clip_boxes(boxes, width, height):
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width  - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width  - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


def box_transform(windows, targets):
    # targets : ground truth

    bw = windows[:, 2] - windows[:, 0] + 1.0
    bh = windows[:, 3] - windows[:, 1] + 1.0
    bx = windows[:, 0] + 0.5 *bw
    by = windows[:, 1] + 0.5 *bh

    tw = targets[:, 2] - targets[:, 0] + 1.0
    th = targets[:, 3] - targets[:, 1] + 1.0
    tx = targets[:, 0] + 0.5 * tw
    ty = targets[:, 1] + 0.5 * th

    dx = (tx - bx) / bw
    dy = (ty - by) / bh
    dw = np.log(tw / bw)
    dh = np.log(th / bh)

    deltas = np.vstack((dx, dy, dw, dh)).transpose()
    return deltas


def box_transform_inv(windows, deltas):

    num   = len(windows)
    predictions = np.zeros((num,4), dtype=np.float32)
    # if num == 0: return predictions  #not possible?

    bw = windows[:, 2] - windows[:, 0] + 1.0
    bh = windows[:, 3] - windows[:, 1] + 1.0
    bx = windows[:, 0] + 0.5 * bw
    by = windows[:, 1] + 0.5 * bh
    bw = bw[:, np.newaxis]
    bh = bh[:, np.newaxis]
    bx = bx[:, np.newaxis]
    by = by[:, np.newaxis]

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    x = dx * bw + bx
    y = dy * bh + by
    w = np.exp(dw) * bw
    h = np.exp(dh) * bh

    predictions[:, 0::4] = x - 0.5 * w  # x0,y0,x1,y1
    predictions[:, 1::4] = y - 0.5 * h
    predictions[:, 2::4] = x + 0.5 * w
    predictions[:, 3::4] = y + 0.5 * h

    return predictions


def filter_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def is_small_box_at_boundary(box,W,H, min_size):
    x0,y0,x1,y1 = box
    w = (x1-x0)+1
    h = (y1-y0)+1
    aspect = max(w,h)/min(w,h)
    area = w*h
    return ((x0==0 or x1==W-1) and (w<min_size)) or \
           ((y0==0 or y1==H-1) and (h<min_size))



def is_small_box(box, min_size):
    x0,y0,x1,y1 = box
    w = (x1-x0)+1
    h = (y1-y0)+1
    aspect = max(w,h)/min(w,h)
    area = w*h
    return (w <min_size or h<min_size)

def is_big_box(box, max_size):
    x0,y0,x1,y1 = box
    w = (x1-x0)+1
    h = (y1-y0)+1
    aspect = max(w,h)/min(w,h)
    area = w*h
    return (w>max_size or h>max_size)





## check  ##############################################################################

def run_check_nms():

    #test nms:
    H,W = 480,640
    num_objects = 4
    rois = []
    for n in range(num_objects):
        w = np.random.randint(64,256)
        h = np.random.randint(64,256)
        x0 = np.random.randint(0,W-w)
        y0 = np.random.randint(0,H-h)
        x1 = x0 + w
        y1 = y0 + h
        gt = [x0,y0,x1,y1]

        M = np.random.randint(10,20)
        for m in range(M):
            dw = int(np.random.uniform(0.5,2)*w)
            dh = int(np.random.uniform(0.5,2)*h)
            dx = int(np.random.uniform(-1,1)*w*0.5)
            dy = int(np.random.uniform(-1,1)*h*0.5)
            xx0 = x0 - dw//2 + dx
            yy0 = y0 - dh//2 + dy
            xx1 = xx0 + w+dw
            yy1 = yy0 + h+dh
            score = np.random.uniform(0.5,2)

            rois.append([xx0,yy0,xx1,yy1, score])
            pass

    rois = np.array(rois).astype(np.float32)


    if 1:
        keep = gpu_nms(rois, 0.5)
        print('gpu_nms    :', keep)
        keep = cython_nms(rois, 0.5)
        print('cython_nms :', keep)

        #gpu     [52, 39, 48, 43, 47, 21, 9, 6, 32, 8, 16, 36, 28, 29, 53, 41]
        #py      [52, 39, 48, 43, 47, 21, 9, 6, 32, 8, 16, 36, 28, 29, 53, 41]
        #cython  [52, 39, 48, 43, 47, 21, 9, 6, 32, 8, 16, 36, 28, 29, 53, 41]

    if 1:
        rois = torch.from_numpy(rois).cuda()
        keep = torch_nms(rois, 0.5)

        keep = keep.cpu().numpy()
        print('torch_nms  :', keep)
        #torch [52 39 48 43 47 21  9  6 32  8 16 36 28 29 53 41]


# # main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_nms()

    print('sucess!')
