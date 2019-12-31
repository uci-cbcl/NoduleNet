import sys
sys.path.append('./')
import numpy as np
import scipy.ndimage
from skimage import measure, morphology
import SimpleITK as sitk
from multiprocessing import Pool
import os
import nrrd
from scipy.ndimage.measurements import label
from config import config


def load_itk_image(filename):
    """Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def binarize(image, spacing, intensity_thred=-600, sigma=1.0, area_thred=30.0,
             eccen_thred=0.99, corner_side=10):
    """
    Binarize the raw 3D CT image slice by slice
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    spacing: float * 3, raw CT spacing in [z, y, x] order.
    intensity_thred: float, thredhold for lung and air
    sigma: float, standard deviation used for Gaussian filter smoothing.
    area_thred: float, min threshold area measure in mm.
    eccen_thred: float, eccentricity thredhold measure how round is an ellipse
    corner_side: int, side length of top-left corner in each slice,
        in terms of pixels.

    return: binary mask with the same shape of the image, that only region of
        interest is True.
    """
    binary_mask = np.zeros(image.shape, dtype=bool)
    side_len = image.shape[1]  # side length of each slice, e.g. 512

    # [-side_len/2, side_len/2], e.g. [-255.5, -254.5, ..., 254.5, 255.5]
    grid_axis = np.linspace(-side_len / 2 + 0.5, side_len / 2 - 0.5, side_len)

    x, y = np.meshgrid(grid_axis, grid_axis)

    #  pixel distance from each pixel to the origin of the slice of shape
    #  [side_len, side_len]
    distance = np.sqrt(np.square(x) + np.square(y))

    # four corners are 0, elsewhere are 1
    nan_mask = (distance < side_len / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan  # assing 0 to be np.nan

    # binarize each slice
    for i in range(image.shape[0]):
        slice_raw = np.array(image[i]).astype('float32')

        # number of differnet values in the top-left corner
        num_uniq = len(np.unique(slice_raw[0:corner_side, 0:corner_side]))

        # black corners out-of-scan, make corners nan before Gaussian filtering
        # (to make corners False in mask)
        if num_uniq == 1:
            slice_raw *= nan_mask

        slice_smoothed = scipy.ndimage.gaussian_filter(slice_raw, sigma,
                                                       truncate=2.0)

        # mask of low-intensity pixels (True = lungs, air)
        slice_binary = slice_smoothed < intensity_thred

        # get connected componets annoated by label
        label = measure.label(slice_binary)
        properties = measure.regionprops(label)
        label_valid = set()

        for prop in properties:
            # area of each componets measured in mm
            area_mm = prop.area * spacing[1] * spacing[2]

            # only include comppents with curtain min area and round enough
            if area_mm > area_thred and prop.eccentricity < eccen_thred:
                label_valid.add(prop.label)

        # test each pixel in label is in label_valid or not and add those True
        # into slice_binary
        slice_binary = np.in1d(label, list(label_valid)).reshape(label.shape)
        binary_mask[i] = slice_binary

    return binary_mask


def exclude_corner_middle(label):
    """
    Exclude componets that are connected to the 8 corners and the middle of
        the 3D image
    label: 3D numpy array of connected component labels with same shape of the
        raw CT image

    return: label after setting those components to 0
    """
    # middle of the left and right lungs
    mid = int(label.shape[2] / 2)

    corner_label = set([label[0, 0, 0],
                        label[0, 0, -1],
                        label[0, -1, 0],
                        label[0, -1, -1],
                        label[-1, 0, 0],
                        label[-1, 0, -1],
                        label[-1, -1, 0],
                        label[-1, -1, -1]])

    middle_label = set([label[0, 0, mid],
                        label[0, -1, mid],
                        label[-1, 0, mid],
                        label[-1, -1, mid]])

    for l in corner_label:
        label[label == l] = 0

    for l in middle_label:
        label[label == l] = 0

    return label


def volume_filter(label, spacing, vol_min=0.2, vol_max=8.2):
    """
    Remove volumes too large/small to be lungs takes out most of air around
    body.
    adult M total lung capacity is 6 L (3L each)
    adult F residual volume is 1.1 L (0.55 L each)
    label: 3D numpy array of connected component labels with same shape of the
        raw CT image.
    spacing: float * 3, raw CT spacing in [z, y, x] order.
    vol_min: float, min volume of the lung
    vol_max: float, max volume of the lung
    """
    properties = measure.regionprops(label)

    for prop in properties:
        if prop.area * spacing.prod() < vol_min * 1e6 or \
           prop.area * spacing.prod() > vol_max * 1e6:
            label[label == prop.label] = 0

    return label


def exclude_air(label, spacing, area_thred=3e3, dist_thred=62):
    """
    Select 3D components that contain slices with sufficient area that,
    on average, are close to the center of the slice. Select component(s) that
    passes this condition:
    1. each slice of the component has significant area (> area_thred),
    2. average min-distance-from-center-pixel < dist_thred
    should select lungs, which are closer to center than out-of-body spaces
    label: 3D numpy array of connected component labels with same shape of the
        raw CT image.
    spacing: float * 3, raw CT spacing in [z, y, x] order.
    area_thred: float, sufficient area
    dist_thred: float, sufficient close
    return: binary mask with the same shape of the image, that only region of
        interest is True. has_lung means if the remaining 3D component is lung
        or not
    """
    # prepare a slice map of distance to center
    y_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5,
                         label.shape[1]) * spacing[1]
    x_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5,
                         label.shape[2]) * spacing[2]
    y, x = np.meshgrid(y_axis, x_axis)

    # real distance from each pixel to the origin of a slice
    distance = np.sqrt(np.square(y) + np.square(x))
    distance_max = np.max(distance)

    # properties of each 3D componet.
    vols = measure.regionprops(label)
    label_valid = set()

    for vol in vols:
        # 3D binary matrix, only voxels within label matches vol.label is True
        single_vol = (label == vol.label)

        # measure area and min_dist for each slice
        # min_distance: distance of closest voxel to center
        # (else max(distance))
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])

        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * distance +
                                     (1 - single_vol[i]) * distance_max)

            # 1. each slice of the component has enough area (> area_thred)
            # 2. average min-distance-from-center-pixel < dist_thred
            if np.average([min_distance[i] for i in range(label.shape[0])
                          if slice_area[i] > area_thred]) < dist_thred:
                label_valid.add(vol.label)

    binary_mask = np.in1d(label, list(label_valid)).reshape(label.shape)
    has_lung = len(label_valid) > 0

    return binary_mask, has_lung


def fill_hole(binary_mask):
    """
    Fill in 3D holes. Select every component that isn't touching corners.
    binary_mask: 3D binary numpy array with the same shape of the image,
        that only region of interest is True.
    """
    # 3D components that are not ROI
    label = measure.label(~binary_mask)

    # idendify corner components
    corner_label = set([label[0, 0, 0],
                        label[0, 0, -1],
                        label[0, -1, 0],
                        label[0, -1, -1],
                        label[-1, 0, 0],
                        label[-1, 0, -1],
                        label[-1, -1, 0],
                        label[-1, -1, -1]])
    binary_mask = ~np.in1d(label, list(corner_label)).reshape(label.shape)

    return binary_mask


def extract_main(binary_mask, cover=0.95):
    """
    Extract lung without bronchi/trachea. Remove small components
    binary_mask: 3D binary numpy array with the same shape of the image,
        that only region of interest is True. One side of the lung in this
        specifical case.
    cover: float, percetange of the total area to keep of each slice, by
        keeping the total connected components
    return: binary mask with the same shape of the image, that only region of
        interest is True. One side of the lung in this specifical case.
    """

    for i in range(binary_mask.shape[0]):
        slice_binary = binary_mask[i]
        label = measure.label(slice_binary)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        areas = [prop.area for prop in properties]
        count = 0
        area_sum = 0
        area_cover = np.sum(areas) * cover

        # count how many components covers, e.g 95%, of total area (lung)
        while area_sum < area_cover:
            area_sum += areas[count]
            count += 1

        # SLICE-WISE: exclude trachea/bronchi.
        # only keep pixels in convex hull of big components, since
        # convex hull contains small components of lungs we still want
        slice_filter = np.zeros(slice_binary.shape, dtype=bool)
        for j in range(count):
            min_row, min_col, max_row, max_col = properties[j].bbox
            slice_filter[min_row:max_row, min_col:max_col] |= \
                properties[j].convex_image

        binary_mask[i] = binary_mask[i] & slice_filter

    label = measure.label(binary_mask)
    properties = measure.regionprops(label)
    properties.sort(key=lambda x: x.area, reverse=True)
    # VOLUME: Return lung, ie the largest component.
    binary_mask = (label == properties[0].label)

    return binary_mask


def fill_2d_hole(binary_mask):
    """
    Fill in holes of binary single lung slicewise.
    binary_mask: 3D binary numpy array with the same shape of the image,
        that only region of interest is True. One side of the lung in this
        specifical case.
    return: binary mask with the same shape of the image, that only region of
        interest is True. One side of the lung in this specifical case.
    """

    for i in range(binary_mask.shape[0]):
        slice_binary = binary_mask[i]
        label = measure.label(slice_binary)
        properties = measure.regionprops(label)

        for prop in properties:
            min_row, min_col, max_row, max_col = prop.bbox
            slice_binary[min_row:max_row, min_col:max_col] |= \
                prop.filled_image  # 2D component without holes

        binary_mask[i] = slice_binary

    return binary_mask


def seperate_two_lung(binary_mask, spacing, max_iter=22, max_ratio=4.8):
    """
    Gradually erode binary mask until lungs are in two separate components
    (trachea initially connects them into 1 component) erosions are just used
    for distance transform to separate full lungs.
    binary_mask: 3D binary numpy array with the same shape of the image,
        that only region of interest is True.
    spacing: float * 3, raw CT spacing in [z, y, x] order.
    max_iter: max number of iterations for erosion.
    max_ratio: max ratio allowed between the volume differences of two lungs
    return: two 3D binary numpy array with the same shape of the image,
        that only region of interest is True. Each binary mask is ROI of one
        side of the lung.
    """
    found = False
    iter_count = 0
    binary_mask_full = np.copy(binary_mask)

    while not found and iter_count < max_iter:
        label = measure.label(binary_mask, connectivity=2)
        properties = measure.regionprops(label)
        # sort componets based on their area
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and \
                properties[0].area / properties[1].area < max_ratio:
            found = True
            # binnary mask for the larger eroded lung
            eroded1 = (label == properties[0].label)
            # binnary mask for the smaller eroded lung
            eroded2 = (label == properties[1].label)
        else:
            # erode the convex hull of each 3D component by 1 voxel
            binary_mask = scipy.ndimage.binary_erosion(binary_mask)
            iter_count += 1

    # because eroded lung will has smaller volums than the original lung,
    # we need to label those eroded voxel based on their distances to the
    # two eroded lungs.
    if found:
        # distance1 has the same shape as the 3D CT image, each voxel contains
        # the euclidient distance from the voxel to the closest voxel within
        # eroded1, so voxel within eroded1 will has distance 0.
        distance1 = scipy.ndimage.morphology.\
            distance_transform_edt(~eroded1, sampling=spacing)
        distance2 = scipy.ndimage.morphology.\
            distance_transform_edt(~eroded2, sampling=spacing)

        # Original mask & lung1 mask
        binary_mask1 = binary_mask_full & (distance1 < distance2)
        # Original mask & lung2 mask
        binary_mask2 = binary_mask_full & (distance1 > distance2)

        # remove bronchi/trachea and other small components
        binary_mask1 = extract_main(binary_mask1)
        binary_mask2 = extract_main(binary_mask2)
    else:
        # did not seperate the two lungs, use the original lung as one of them
        binary_mask1 = binary_mask_full
        binary_mask2 = np.zeros(binary_mask.shape).astype('bool')

    binary_mask1 = fill_2d_hole(binary_mask1)
    binary_mask2 = fill_2d_hole(binary_mask2)

    return binary_mask1, binary_mask2


def extract_lung(image, spacing):
    """
    Preprocess pipeline for extracting the lung from the raw 3D CT image.
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    spacing: float * 3, raw CT spacing in [z, y, x] order.
    return: two 3D binary numpy array with the same shape of the image,
        that only region of interest is True. Each binary mask is ROI of one
        side of the lung. Also return if lung is found or not.
    """
    # binary mask with the same shape of the image, that only region of
    # interest is True.
    binary_mask = binarize(image, spacing)

    # labelled 3D connected componets, with the same shape as image. each
    # commponet has a different int number > 0
    label = measure.label(binary_mask, connectivity=1)

    # exclude componets that are connected to the 8 corners and the middle
    # of the 3D image
    label = exclude_corner_middle(label)

    # exclude componets that are too small or too large to be lung
    label = volume_filter(label, spacing)

    # exclude more air chunks and grab lung mask region
    binary_mask, has_lung = exclude_air(label, spacing)

    # fill in 3D holes. Select every component that isn't touching corners.
    binary_mask = fill_hole(binary_mask)

    # seperate two lungs
    binary_mask1, binary_mask2 = seperate_two_lung(binary_mask, spacing)

    return (binary_mask1, binary_mask2, has_lung)


def HU2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype('uint8')

    return image_new


def convex_hull_dilate(binary_mask, dilate_factor=1.5, iterations=10):
    """
    Replace each slice with convex hull of it then dilate. Convex hulls used
    only if it does not increase area by dilate_factor. This applies mainly to
    the inferior slices because inferior surface of lungs is concave.
    binary_mask: 3D binary numpy array with the same shape of the image,
        that only region of interest is True. One side of the lung in this
        specifical case.
    dilate_factor: float, factor of increased area after dilation
    iterations: int, number of iterations for dilation
    return: 3D binary numpy array with the same shape of the image,
        that only region of interest is True. Each binary mask is ROI of one
        side of the lung.
    """
    binary_mask_dilated = np.array(binary_mask)
    for i in range(binary_mask.shape[0]):
        slice_binary = binary_mask[i]

        if np.sum(slice_binary) > 0:
            slice_convex = morphology.convex_hull_image(slice_binary)

            if np.sum(slice_convex) <= dilate_factor * np.sum(slice_binary):
                binary_mask_dilated[i] = slice_convex

    struct = scipy.ndimage.morphology.generate_binary_structure(3, 1)
    binary_mask_dilated = scipy.ndimage.morphology.binary_dilation(
        binary_mask_dilated, structure=struct, iterations=10)

    return binary_mask_dilated


def apply_mask(image, binary_mask1, binary_mask2, pad_value=170,
               bone_thred=210, remove_bone=False):
    """
    Apply the binary mask of each lung to the image. Regions out of interest
    are replaced with pad_value.
    image: 3D uint8 numpy array with the same shape of the image.
    binary_mask1: 3D binary numpy array with the same shape of the image,
        that only one side of lung is True.
    binary_mask2: 3D binary numpy array with the same shape of the image,
        that only the other side of lung is True.
    pad_value: int, uint8 value for padding image regions that is not
        interested.
    bone_thred: int, uint8 threahold value for determine parts of image is
        bone.
    return: D uint8 numpy array with the same shape of the image after
        applying the lung mask.
    """
    binary_mask = binary_mask1 + binary_mask2
    binary_mask1_dilated = convex_hull_dilate(binary_mask1)
    binary_mask2_dilated = convex_hull_dilate(binary_mask2)
    binary_mask_dilated = binary_mask1_dilated + binary_mask2_dilated
    binary_mask_extra = binary_mask_dilated ^ binary_mask

    # replace image values outside binary_mask_dilated as pad value
    image_new = image * binary_mask_dilated + \
        pad_value * (1 - binary_mask_dilated).astype('uint8')

    # set bones in extra mask to 170 (ie convert HU > 482 to HU 0;
    # water).
    if remove_bone:
        image_new[image_new * binary_mask_extra > bone_thred] = pad_value

    return image_new


def resample(image, spacing, new_spacing=[1.0, 1.0, 1.0], order=1):
    """
    Resample image from the original spacing to new_spacing, e.g. 1x1x1
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    spacing: float * 3, raw CT spacing in [z, y, x] order.
    new_spacing: float * 3, new spacing used for resample, typically 1x1x1,
        which means standardizing the raw CT with different spacing all into
        1x1x1 mm.
    order: int, order for resample function scipy.ndimage.interpolation.zoom
    return: 3D binary numpy array with the same shape of the image after,
        resampling. The actual resampling spacing is also returned.
    """
    # shape can only be int, so has to be rounded.
    new_shape = np.round(image.shape * spacing / new_spacing)

    # the actual spacing to resample.
    resample_spacing = spacing * image.shape / new_shape

    resize_factor = new_shape / image.shape

    image_new = scipy.ndimage.interpolation.zoom(image, resize_factor,
                                                 mode='nearest', order=order)

    return (image_new, resample_spacing)


def get_lung_box(binary_mask, new_shape, margin=5):
    """
    Get the lung barely surrounding the lung based on the binary_mask and the
    new_spacing.
    binary_mask: 3D binary numpy array with the same shape of the image,
        that only region of both sides of the lung is True.
    new_shape: tuple of int * 3, new shape of the image after resamping in
        [z, y, x] order.
    margin: int, number of voxels to extend the boundry of the lung box.
    return: 3x2 2D int numpy array denoting the
        [z_min:z_max, y_min:y_max, x_min:x_max] of the lung box with respect to
        the image after resampling.
    """
    # list of z, y x indexes that are true in binary_mask
    z_true, y_true, x_true = np.where(binary_mask)
    old_shape = binary_mask.shape

    lung_box = np.array([[np.min(z_true), np.max(z_true)],
                        [np.min(y_true), np.max(y_true)],
                        [np.min(x_true), np.max(x_true)]])
    lung_box = lung_box * 1.0 * \
        np.expand_dims(new_shape, 1) / np.expand_dims(old_shape, 1)
    lung_box = np.floor(lung_box).astype('int')

    z_min, z_max = lung_box[0]
    y_min, y_max = lung_box[1]
    x_min, x_max = lung_box[2]

    # extend the lung_box by a margin
    lung_box[0] = max(0, z_min-margin), min(new_shape[0], z_max+margin)
    lung_box[1] = max(0, y_min-margin), min(new_shape[1], y_max+margin)
    lung_box[2] = max(0, x_min-margin), min(new_shape[2], x_max+margin)

    return lung_box


def auxiliary_segment(image):
    """
    In case of failure of the first segmentation method, use sitk lib for further segmentation
    image: numpy array of raw CT image, [D, H, W] in z, y, x order
    return: numpy array of lung mask
    """
    def fill_hole_2d(image):
        """
        Fill hole slice by slice from axial view
        image: numpy array of raw CT image, [D, H, W]
        """
        image = image.copy()
        D, H, W = image.shape
        sitk_img = sitk.GetImageFromArray(image)
        
        for i in range(D):
            image[i] = sitk.GetArrayFromImage(sitk.BinaryFillhole(sitk_img[:, :, i]))
            
        return sitk.GetImageFromArray(image)

    def morphology_closing_2d(image):
        """
        Morphology closing slice by slice from axial view
        image: numpy array of raw CT image, [D, H, W]
        """
        image = image.copy()
        D, H, W = image.shape
        sitk_img = sitk.GetImageFromArray(image)
        
        for i in range(D):
            image[i] = sitk.GetArrayFromImage(sitk.BinaryMorphologicalOpening(sitk_img[:, :, i], 5))
            
        return sitk.GetImageFromArray(image)

    mask = 1 - sitk.OtsuThreshold(sitk.GetImageFromArray(image))
    mask = morphology_closing_2d(sitk.GetArrayFromImage(mask))

    mask_npy = sitk.GetArrayFromImage(mask)
    chest_mask = fill_hole_2d(mask_npy)
    lung_mask = sitk.Subtract(sitk.GetImageFromArray(sitk.GetArrayFromImage(chest_mask)), mask)
        
    # Remove areas not in the chest, when CT covers regions below the chest
    eroded_mask = sitk.BinaryErode(lung_mask, 15)
    seed_npy = sitk.GetArrayFromImage(eroded_mask)
    seed_npy = np.array(seed_npy.nonzero())[[2,1,0]]
    seeds = seed_npy.T.tolist()
    connected_lung = sitk.ConfidenceConnected(lung_mask, seeds, multiplier=2.5)
    final_mask = sitk.BinaryMorphologicalClosing(connected_lung, 5)
    final_mask = sitk.BinaryDilate(final_mask, 5)
    
    return sitk.GetArrayFromImage(final_mask)


def preprocess(params):
    pid, lung_mask_dir, nod_mask_dir, img_dir, save_dir, do_resample = params
    
    print('Preprocessing %s...' % (pid))

    lung_mask, _, _ = load_itk_image(os.path.join(lung_mask_dir, '%s.mhd' % (pid)))
    img, origin, spacing = load_itk_image(os.path.join(img_dir, '%s.mhd' % (pid)))
    nod_mask, _ = nrrd.read(os.path.join(nod_mask_dir, '%s' % (pid)))

    binary_mask1, binary_mask2 = lung_mask == 4, lung_mask == 3
    binary_mask = binary_mask1 + binary_mask2

    img = HU2uint8(img)
    seg_img = apply_mask(img, binary_mask1, binary_mask2)

    if do_resample:
        print('Resampling...')
        seg_img, resampled_spacing = resample(seg_img, spacing, order=3)
        seg_nod_mask = np.zeros(seg_img.shape, dtype=np.uint8)
        for i in range(int(nod_mask.max())):
            mask = (nod_mask == (i + 1)).astype(np.uint8)
            mask, _ = resample(mask, spacing, order=3)
            seg_nod_mask[mask > 0.5] = i + 1

    lung_box = get_lung_box(binary_mask, seg_img.shape)

    z_min, z_max = lung_box[0]
    y_min, y_max = lung_box[1]
    x_min, x_max = lung_box[2]

    seg_img = seg_img[z_min:z_max, y_min:y_max, x_min:x_max]
    seg_nod_mask = seg_nod_mask[z_min:z_max, y_min:y_max, x_min:x_max]
    np.save(os.path.join(save_dir, '%s_origin.npy' % (pid)), origin)
    np.save(os.path.join(save_dir, '%s_spacing.npy' % (pid)), resampled_spacing)
    np.save(os.path.join(save_dir, '%s_ebox_origin.npy' % (pid)), np.array((z_min, y_min, x_min)))
    nrrd.write(os.path.join(save_dir, '%s_clean.nrrd' % (pid)), seg_img)
    nrrd.write(os.path.join(save_dir, '%s_mask.nrrd' % (pid)), seg_nod_mask)

    print('number of nodules before: %s, afeter preprocessing: %s' % (nod_mask.max(), seg_nod_mask.max()))
    print('Finished %s' % (pid))
    print()


def generate_label(params):
    pid, lung_mask_dir, nod_mask_dir, img_dir, save_dir, do_resample = params
    masks, _ = nrrd.read(os.path.join(save_dir, '%s_mask.nrrd' % (pid)))

    bboxes = []
    instance_nums = [num for num in np.unique(masks) if num]
    for i in instance_nums:
        mask = (masks == i).astype(np.uint8)
        zz, yy, xx = np.where(mask)
        d = max(zz.max() - zz.min() + 1,  yy.max() - yy.min() + 1, xx.max() - xx.min() + 1)
        bboxes.append(np.array([(zz.max() + zz.min()) / 2., (yy.max() + yy.min()) / 2., (xx.max() + xx.min()) / 2., d]))
        
    bboxes = np.array(bboxes)
    if not len(bboxes):
        print('%s does not have any nodules!!!' % (pid))

    print('Finished masks to bboxes %s' % (pid))

    np.save(os.path.join(save_dir, '%s_bboxes.npy' % (pid)), bboxes)

def main():
    n_consensus = 3
    do_resample = True
    lung_mask_dir = config['lung_mask_dir']
    nod_mask_dir = os.path.join(config['mask_save_dir'], str(n_consensus))
    img_dir = config['data_dir']
    save_dir = os.path.join(config['preprocessed_data_dir'])
    print('nod mask dir', nod_mask_dir)
    print('save dir ', save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    params_lists = []
    for pid in os.listdir(nod_mask_dir):
        params_lists.append([pid, lung_mask_dir, nod_mask_dir, img_dir, save_dir, do_resample])
    
    pool = Pool(processes=10)
    pool.map(preprocess, params_lists)
    
    pool.close()
    pool.join()

    pool = Pool(processes=10)
    pool.map(generate_label, params_lists)
    
    pool.close()
    pool.join()


if __name__=='__main__':
    main()
