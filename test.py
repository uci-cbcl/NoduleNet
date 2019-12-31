import numpy as np
import torch
import os
import traceback
import time
import nrrd
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
import SimpleITK as sitk
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage.measurements import label
from scipy.ndimage import center_of_mass
from net.nodule_net import NoduleNet
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from dataset.mask_reader import MaskReader
from config import config
from utils.visualize import draw_gt, draw_pred, generate_image_anim
from utils.util import dice_score_seperate, get_contours_from_masks, merge_contours, hausdorff_distance
from utils.util import onehot2multi_mask, normalize, pad2factor, load_dicom_image, crop_boxes2mask_single, npy2submission
import pandas as pd
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

plt.rcParams['figure.figsize'] = (24, 16)
plt.switch_backend('agg')
this_module = sys.modules[__name__]
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


parser = argparse.ArgumentParser()
parser.add_argument('--net', '-m', metavar='NET', default=config['net'],
                    help='neural net')
parser.add_argument("mode", type=str,
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default=config['initial_checkpoint'],
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test-set-name", type=str, default=config['test_set_name'],
                    help="path to save the results")


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()
    # params_eye_L = np.load('weights/params_eye_L.npy').item()
    # params_eye_R = np.load('weights/params_eye_R.npy').item()
    # params_brain_stem = np.load('weights/params_brain_stem.npy').item()

    if args.mode == 'eval':
        data_dir = config['preprocessed_data_dir']
        test_set_name = args.test_set_name
        num_workers = 0
        initial_checkpoint = args.weight
        net = args.net
        out_dir = args.out_dir

        net = getattr(this_module, net)(config)
        net = net.cuda()

        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            # out_dir = checkpoint['out_dir']
            epoch = checkpoint['epoch']

            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('No model weight file specified')
            return

        print('out_dir', out_dir)
        save_dir = os.path.join(out_dir, 'res', str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'FROC')):
            os.makedirs(os.path.join(save_dir, 'FROC'))

        dataset = MaskReader(data_dir, test_set_name, config, mode='eval')
        eval(net, dataset, save_dir)
    else:
        logging.error('Mode %s is not supported' % (args.mode))


def eval(net, dataset, save_dir=None):
    net.set_mode('eval')
    net.use_mask = False
    net.use_rcnn = True
    aps = []
    dices = []
    raw_dir = config['data_dir']
    preprocessed_dir = config['preprocessed_data_dir']

    print('Total # of eval data %d' % (len(dataset)))
    for i, (input, truth_bboxes, truth_labels, truth_masks, mask, image) in enumerate(dataset):
        try:
            D, H, W = image.shape
            pid = dataset.filenames[i]

            print('[%d] Predicting %s' % (i, pid), image.shape)
            gt_mask = mask.astype(np.uint8)

            with torch.no_grad():
                input = input.cuda().unsqueeze(0)
                net.forward(input, truth_bboxes, truth_labels, truth_masks, mask)

            rpns = net.rpn_proposals.cpu().numpy()
            detections = net.detections.cpu().numpy()
            ensembles = net.ensemble_proposals.cpu().numpy()

            if len(detections) and net.use_mask:
                crop_boxes = net.crop_boxes
                segments = [F.sigmoid(m).cpu().numpy() > 0.5 for m in net.mask_probs]

                pred_mask = crop_boxes2mask_single(crop_boxes[:, 1:], segments, input.shape[2:])
                pred_mask = pred_mask.astype(np.uint8)

                # compute average precisions
                ap, dice = average_precision(gt_mask, pred_mask)
                aps.append(ap)
                dices.extend(dice.tolist())
                print(ap)
                print('AP: ', np.mean(ap))
                print('DICE: ', dice)
                print
            else:
                pred_mask = np.zeros((input[0].shape))
            
            np.save(os.path.join(save_dir, '%s.npy' % (pid)), pred_mask)

            print('rpn', rpns.shape)
            print('detection', detections.shape)
            print('ensemble', ensembles.shape)


            if len(rpns):
                rpns = rpns[:, 1:]
                np.save(os.path.join(save_dir, '%s_rpns.npy' % (pid)), rpns)

            if len(detections):
                detections = detections[:, 1:-1]
                np.save(os.path.join(save_dir, '%s_rcnns.npy' % (pid)), detections)

            if len(ensembles):
                ensembles = ensembles[:, 1:]
                np.save(os.path.join(save_dir, '%s_ensembles.npy' % (pid)), ensembles)


            # Clear gpu memory
            del input, truth_bboxes, truth_labels, truth_masks, mask, image, pred_mask#, gt_mask, gt_img, pred_img, full, score
            torch.cuda.empty_cache()

        except Exception as e:
            del input, truth_bboxes, truth_labels, truth_masks, mask, image,
            torch.cuda.empty_cache()
            traceback.print_exc()
                        
            print
            return

    aps = np.array(aps)
    dices = np.array(dices)
    print('mAP: ', np.mean(aps, 0))
    print('mean dice:%.4f(%.4f)' % (np.mean(dices), np.std(dices)))
    print('mean dice (exclude fn):%.4f(%.4f)' % (np.mean(dices[dices != 0]), np.std(dices[dices != 0])))
    
    # Generate prediction csv for the use of performning FROC analysis
    # Save both rpn and rcnn results
    rpn_res = []
    rcnn_res = []
    ensemble_res = []
    for pid in dataset.filenames:
        if os.path.exists(os.path.join(save_dir, '%s_rpns.npy' % (pid))):
            rpns = np.load(os.path.join(save_dir, '%s_rpns.npy' % (pid)))
            rpns = rpns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_rcnns.npy' % (pid))):
            rcnns = np.load(os.path.join(save_dir, '%s_rcnns.npy' % (pid)))
            rcnns = rcnns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rcnns))
            rcnn_res.append(np.concatenate([names, rcnns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_ensembles.npy' % (pid))):
            ensembles = np.load(os.path.join(save_dir, '%s_ensembles.npy' % (pid)))
            ensembles = ensembles[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(ensembles))
            ensemble_res.append(np.concatenate([names, ensembles], axis=1))
    
    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    ensemble_res = np.concatenate(ensemble_res, axis=0)
    col_names = ['seriesuid','coordX','coordY','coordZ','diameter_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
    
    df = pd.DataFrame(rpn_res, columns=col_names)
    df.to_csv(rpn_submission_path, index=False)

    df = pd.DataFrame(rcnn_res, columns=col_names)
    df.to_csv(rcnn_submission_path, index=False)

    df = pd.DataFrame(ensemble_res, columns=col_names)
    df.to_csv(ensemble_submission_path, index=False)

    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))
    if not os.path.exists(os.path.join(eval_dir, 'rcnn')):
        os.makedirs(os.path.join(eval_dir, 'rcnn'))
    if not os.path.exists(os.path.join(eval_dir, 'ensemble')):
        os.makedirs(os.path.join(eval_dir, 'ensemble'))

    noduleCADEvaluation('evaluationScript/annotations/LIDC/3_annotation.csv',
    'evaluationScript/annotations/LIDC/3_annotation_excluded.csv',
    dataset.set_name, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    noduleCADEvaluation('evaluationScript/annotations/LIDC/3_annotation.csv',
    'evaluationScript/annotations/LIDC/3_annotation_excluded.csv',
    dataset.set_name, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    noduleCADEvaluation('evaluationScript/annotations/LIDC/3_annotation.csv',
    'evaluationScript/annotations/LIDC/3_annotation_excluded.csv',
    dataset.set_name, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))
        
    print


def eval_single(net, input):
    with torch.no_grad():
        input = input.cuda().unsqueeze(0)
        logits = net.forward(input)
        logits = logits[0]
    
    masks = logits.cpu().data.numpy()
    masks = (masks > 0.5).astype(np.int32)
    return masks
 

if __name__ == '__main__':
    main()
