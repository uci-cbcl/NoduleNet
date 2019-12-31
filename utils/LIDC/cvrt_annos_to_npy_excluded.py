from lidc_nodule_detection.pylung.annotation import *
from tqdm import tqdm
import sys
import nrrd
import SimpleITK as sitk
import cv2
from config import config


def load_itk_image(filename):
    """Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def xml2mask(xml_file):
    header, annos = parse(xml_file)

    ctr_arrs = []
    for i, reader in enumerate(annos):
        for j, nodule in enumerate(reader.nodules):
            ctr_arr = []
            for k, roi in enumerate(nodule.rois):
                z = roi.z
                for roi_xy in roi.roi_xy:
                    ctr_arr.append([z, roi_xy[1], roi_xy[0]])
            ctr_arrs.append(ctr_arr)
            
    seriesuid = header.series_instance_uid
    return seriesuid, ctr_arrs


def annotation2masks(annos_dir, save_dir):
    files = find_all_files(annos_dir, '.xml')
    for f in tqdm(files, total=len(files)):
        try:
            seriesuid, masks = xml2mask(f)
            np.save(os.path.join(save_dir, '%s' % (seriesuid)), masks)
        except:
            print("Unexpected error:", sys.exc_info()[0])
    
def arr2mask(arr, reso):
    mask = np.zeros(reso)
    arr = arr.astype(np.int32)
    mask[arr[:, 0], arr[:, 1], arr[:, 2]] = 1
    
    return mask

def arrs2mask(img_dir, ctr_arr_dir, save_dir):
    pids = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.mhd')]
    cnt = 0
    consensus = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for k in consensus.keys():
        if not os.path.exists(os.path.join(save_dir, str(k))):
            os.makedirs(os.path.join(save_dir, str(k)))

    for pid in tqdm(pids, total=len(pids)):
        img, origin, spacing = load_itk_image(os.path.join(img_dir, '%s.mhd' % (pid)))
        ctr_arrs = np.load(os.path.join(ctr_arr_dir, '%s.npy' % (pid)))
        cnt += len(ctr_arrs)

        nodule_masks = []
        for ctr_arr in ctr_arrs:
            z_origin = origin[0]
            z_spacing = spacing[0]
            ctr_arr = np.array(ctr_arr)
            ctr_arr[:, 0] = np.absolute(ctr_arr[:, 0] - z_origin) / z_spacing
            ctr_arr = ctr_arr.astype(np.int32)

            mask = np.zeros(img.shape)

            for z in np.unique(ctr_arr[:, 0]):
                ctr = ctr_arr[ctr_arr[:, 0] == z][:, [2, 1]]
                ctr = np.array([ctr], dtype=np.int32)
                mask[z] = cv2.fillPoly(mask[z], ctr, color=(1,) * 1)
            nodule_masks.append(mask)

        i = 0
        visited = []
        d = {}
        masks = []
        while i < len(nodule_masks):
            # If mached before, then no need to create new mask
            if i in visited:
                i += 1
                continue
            same_nodules = []
            mask1 = nodule_masks[i]
            same_nodules.append(mask1)
            d[i] = {}
            d[i]['count'] = 1
            d[i]['iou'] = []

            # Find annotations pointing to the same nodule
            for j in range(i + 1, len(nodule_masks)):
                # if not overlapped with previous added nodules
                if j in visited:
                    continue
                mask2 = nodule_masks[j]
                iou = float(np.logical_and(mask1, mask2).sum()) / np.logical_or(mask1, mask2).sum()

                if iou > 0.4:
                    visited.append(j)
                    same_nodules.append(mask2)
                    d[i]['count'] += 1
                    d[i]['iou'].append(iou)

            masks.append(same_nodules)
            i += 1

        for k, v in d.iteritems():
            if v['count'] > 4:
                print('WARNING:  %s: %dth nodule, iou: %s' % (pid, k, str(v['iou'])))
                v['count'] = 4
            consensus[v['count']] += 1

        # number of consensus
        num = np.array([len(m) for m in masks])
        num[num > 4] = 4
        
        if len(num) == 0:
            continue
        # Iterate from the nodules with most consensus
        for n in range(1, num.max() + 1):
            mask = np.zeros(img.shape, dtype=np.uint8)
            
            for i, index in enumerate(np.where(num < n)[0]):
                same_nodules = masks[index]
                m = np.logical_or.reduce(same_nodules)
                mask[m] = i + 1
            nrrd.write(os.path.join(save_dir, str(n), pid), mask)
        
#         for i, same_nodules in enumerate(masks):
#             cons = len(same_nodules)
#             if cons > 4:
#                 cons = 4
#             m = np.logical_or.reduce(same_nodules)
#             mask[m] = i + 1
#             nrrd.write(os.path.join(save_dir, str(cons), pid), mask)
        
    print consensus
    print cnt

if __name__ == '__main__':
    annos_dir = config['annos_dir']
    img_dir = config['data_dir']
    ctr_arr_save_dir = config['ctr_arr_save_dir']
    mask_save_dir = config['mask_exclude_save_dir']
    annotation2masks(annos_dir, ctr_arr_save_dir)
    arrs2mask(img_dir, ctr_arr_save_dir, mask_save_dir)
