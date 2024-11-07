# demo
import cv2
import numpy as np
from skimage import io
import glob

# MAE, Precision, Recall, F-measure, IoU, Precision-Recall curves
import numpy as np
from skimage import io
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from . import misc

def mask_normalize(mask):
# input 'mask': HxW
# output: HxW [0,255]
    return mask/(np.amax(mask)+1e-8)

def mask_iou(pred_label,label):
    '''
    calculate mask iou for pred_label and gt_label
    '''

    pred_label = (pred_label>0)[0].int()
    label = (label>0)[0].int()

    intersection = ((label * pred_label) > 0).sum()
    union = ((label + pred_label) > 0).sum()
    return intersection / union


def mask_iter_union(pred_label,label):
    '''
    calculate mask iou for pred_label and gt_label
    '''
    pred_label = (pred_label>0)[0].int()
    label = (label>0)[0].int()

    intersection = ((label * pred_label) > 0).sum()
    union = ((label + pred_label) > 0).sum()
    return intersection, union


# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    device = gt.device
    dt = (dt>0)[0].cpu().byte().numpy()
    gt = (gt>128)[0].cpu().byte().numpy()

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return torch.tensor(boundary_iou).float().to(device)

def compute_iou(preds, target, return_inter_union=False):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds

    if return_inter_union :
        inter_all, union_all = 0, 0
        for i in range(0,len(preds)):
            inter, union = mask_iter_union(postprocess_preds[i],target[i])
            inter_all += inter
            union_all += union
        return inter_all, union_all
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

# def compute_mae(preds, target):
#     assert target.shape[1] == 1, 'only support one mask per image now'
#     if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
#         postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
#     else:
#         postprocess_preds = preds
    
#     mae = torch.abs(postprocess_preds - target).float().mean() 

#     return mae

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def compute_mae(mask1,mask2):
# input 'mask1': HxW or HxWxn (asumme that all the n channels are the same and only the first channel will be used)
#       'mask2': HxW or HxWxn
# output: a value MAE, Mean Absolute Error
    if(len(mask1.shape)<2 or len(mask2.shape)<2):
        print("ERROR: Mask1 or mask2 is not matrix!")
        exit()
    if(len(mask1.shape)>2):
        mask1 = mask1[:,:,0]
    if(len(mask2.shape)>2):
        mask2 = mask2[:,:,0]
    if(mask1.shape!=mask2.shape):
        print("ERROR: The shapes of mask1 and mask2 are different!")
        exit()

    h,w = mask1.shape[0],mask1.shape[1]
    mask1 = mask_normalize(mask1)
    mask2 = mask_normalize(mask2)
    sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    maeError = sumError/(float(h)*float(w)+1e-8)

    return maeError

def compute_ave_MAE_of_methods(gts, results):
#input 'gt_name_list': ground truth name list
#input 'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
#output average Mean Absolute Error, 1xN, N is the number of folders
#output 'gt2rs': numpy array with shape of (num_rs_dir)

    num_gt = len(gts) # number of ground truth files
    if(num_gt==0):
        print("ERROR: The ground truth directory is empty!")
        exit()

    mae = np.zeros((num_gt)) # MAE of methods
    gt2rs = np.zeros((num_gt)) # indicate if the mask mae of methods is correctly computed
    for i in range(0, num_gt):
        # print('-Processed %d/%d'%(i+1,num_gt),end='\r')
        #print("Completed {:2.0%}".format(i / num_gt), end="\r") # print percentile of processed, python 3.0 and newer version
        gt = gts[i] 
        rs = results[i] 
        tmp_mae = 0.0
        try:
            tmp_mae = compute_mae(gt,rs) # compute the mae
        except IOError:
            #print('ERROR: Fails in compute_mae!')
            continue
        mae[i] = tmp_mae
        gt2rs[i] = 1.0
    mae_col_sum = np.sum(mae,0) # compute the sum of MAE of each method
    gt2rs = np.sum(gt2rs,0) # compute the number of correctly computed MAE of each method
    ave_maes = mae_col_sum/(gt2rs+1e-8) # compute the average MAE of each method
    return ave_maes, gt2rs

def compute_pre_rec(gt,mask,mybins=np.arange(0,256)):

    if(len(gt.shape)<2 or len(mask.shape)<2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(mask.shape)>2): # convert to one channel
        mask = mask[:,:,0]
    if(gt.shape!=mask.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()

    gtNum = gt[gt>128].size # pixel number of ground truth foreground regions
    pp = mask[gt>128] # mask predicted pixel values in the ground truth foreground region
    nn = mask[gt<=128] # mask predicted pixel values in the ground truth bacground region

    pp_hist,pp_edges = np.histogram(pp,bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist,nn_edges = np.histogram(nn,bins=mybins)

    pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)

    precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
    recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)

    precision[np.isnan(precision)]= 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision,(len(precision))),np.reshape(recall,(len(recall)))

def compute_PRE_REC_FM_of_methods(gts, results, beta=0.3):
#input 'gt_name_list': ground truth name list
#input 'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
#output precision 'PRE': numpy array with shape of (num_rs_dir, 256)
#       recall    'REC': numpy array with shape of (num_rs_dir, 256)
#       F-measure (beta) 'FM': numpy array with shape of (num_rs_dir, 256)

    mybins = np.arange(0,256) # different thresholds to achieve binarized masks for pre, rec, Fm measures

    num_gt = len(gts) # number of ground truth files
    if(num_gt==0):
        #print("ERROR: The ground truth directory is empty!")
        exit()

    PRE = np.zeros((num_gt,len(mybins)-1)) # PRE: with shape of (num_gt, 256)
    REC = np.zeros((num_gt,len(mybins)-1)) # REC: the same shape with PRE
    # FM = np.zeros((num_gt,len(mybins)-1)) # Fm: the same shape with PRE
    gt2rs = np.zeros((num_gt)) # indicate if the mask of methods is correctly computed

    for i in range(0,num_gt):
        # print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
        gt = gts[i] 
        rs = results[i] 
        pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins)) # pre, rec, f or one mask w.r.t different thresholds

        try:
            pre, rec = compute_pre_rec(gt,rs,mybins=np.arange(0,256))
        except IOError:
            #print('ERROR: Fails in compute_mae!')
            continue
        PRE[i,:] = pre
        REC[i,:] = rec
        gt2rs[i] = 1.0
    print('\n')
    gt2rs = np.sum(gt2rs,0) # num_rs_dir
    gt2rs = np.repeat(gt2rs[np.newaxis], 255, axis=0) #num_rs_dirx255

    PRE = np.sum(PRE,0)/(gt2rs+1e-8) # num_rs_dirx255, average PRE over the whole dataset at every threshold
    REC = np.sum(REC,0)/(gt2rs+1e-8) # num_rs_dirx255
    FM = (1+beta)*PRE*REC/(beta*PRE+REC+1e-8) # num_rs_dirx255

    return PRE, REC, FM, gt2rs

def plot_save_pr_curves(PRE, REC, method_names, lineSylClr, linewidth, xrange=(0.0,1.0), yrange=(0.0,1.0), dataset_name = 'TEST', save_dir = './', save_fmt = 'pdf'):
    fig1 = plt.figure(1)
    num = PRE.shape[0]
    for i in range(0,num):
        if (len(np.array(PRE[i]).shape)!=0):
            print(REC[i].shape)
            plt.plot(REC[i], PRE[i],lineSylClr[i],linewidth=linewidth[i],label=method_names[i])
    # plt.plot(REC, PRE,lineSylClr,linewidth=linewidth,label=method_names)

    plt.xlim(xrange[0],xrange[1])
    plt.ylim(yrange[0],yrange[1])

    xyrange1 = np.arange(xrange[0],xrange[1]+0.01,0.1)
    xyrange2 = np.arange(yrange[0],yrange[1]+0.01,0.1)

    plt.tick_params(direction='in')
    plt.xticks(xyrange1,fontsize=15,fontname='serif')
    plt.yticks(xyrange2,fontsize=15,fontname='serif')

    ## draw dataset name
    plt.text((xrange[0]+xrange[1])/2.0,yrange[0]+0.02,dataset_name,horizontalalignment='center',fontsize=20, fontname='serif',fontweight='bold')

    plt.xlabel('Recall',fontsize=20,fontname='serif')
    plt.ylabel('Precision',fontsize=20,fontname='serif')

    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 7,
    }

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [len(handles)-x for x in range(1,len(handles)+1)]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='lower left', prop=font1)
    plt.grid(linestyle='--')
    fig1.savefig(save_dir+dataset_name+"_pr_curves."+save_fmt,bbox_inches='tight',dpi=300)
    print('>>PR-curves saved: %s'%(save_dir+dataset_name+"_pr_curves."+save_fmt))

def plot_save_fm_curves(FM, mybins, method_names, lineSylClr, linewidth, xrange=(0.0,1.0), yrange=(0.0,1.0), dataset_name = 'TEST', save_dir = './', save_fmt = 'pdf'):

    fig2 = plt.figure(2)
    num = FM.shape[0]
    for i in range(0,num):
        if (len(np.array(FM[i]).shape)!=0):
            print(FM[i].shape)
            plt.plot(np.array(mybins[0:-1]).astype(np.float16)/255.0, FM[i],lineSylClr[i],linewidth=linewidth[i],label=method_names[i])

    plt.xlim(xrange[0],xrange[1])
    plt.ylim(yrange[0],yrange[1])

    xyrange1 = np.arange(xrange[0],xrange[1]+0.01,0.1)
    xyrange2 = np.arange(yrange[0],yrange[1]+0.01,0.1)

    plt.tick_params(direction='in')
    plt.xticks(xyrange1,fontsize=15,fontname='serif')
    plt.yticks(xyrange2,fontsize=15,fontname='serif')

    ## draw dataset name
    plt.text((xrange[0]+xrange[1])/2.0,yrange[0]+0.02,dataset_name,horizontalalignment='center',fontsize=20, fontname='serif',fontweight='bold')

    plt.xlabel('Thresholds',fontsize=20,fontname='serif')
    plt.ylabel('F-measure',fontsize=20,fontname='serif')

    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 7,
    }

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [len(handles)-x for x in range(1,len(handles)+1)]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='lower left', prop=font1)
    plt.grid(linestyle='--')
    fig2.savefig(save_dir+dataset_name+"_fm_curves."+save_fmt,bbox_inches='tight',dpi=300)
    print('>>F-measure curves saved: %s'%(save_dir+dataset_name+"_fm_curves."+save_fmt))

def eval():
    ## 0. =======set the data path=======
    print("------0. set the data path------")

    # >>>>>>> Follows have to be manually configured <<<<<<< #
    data_name = 'TEST-DATA' # this will be drawn on the bottom center of the figures
    data_dir = './test_data/' # set the data directory,
                            # ground truth and results to-be-evaluated should be in this directory
                            # the figures of PR and F-measure curves will be saved in this directory as well
    gt_dir = 'gt' # set the ground truth folder name
    rs_dirs = ['rs1','rs2'] # set the folder names of different methods
                            # 'rs1' contains the result of method1
                            # 'rs2' contains the result of method 2
                            # we suggest to name the folder as the method names because they will be shown in the figures' legend
    lineSylClr = ['r-','b-'] # curve style, same size with rs_dirs
    linewidth = [2,1] # line width, same size with rs_dirs
    # >>>>>>> Above have to be manually configured <<<<<<< #

    gt_name_list = glob.glob(data_dir+gt_dir+'/'+'*.png') # get the ground truth file name list

    ## get directory list of predicted maps
    rs_dir_lists = []
    for i in range(len(rs_dirs)):
        rs_dir_lists.append(data_dir+rs_dirs[i]+'/')
    print('\n')


    ## 1. =======compute the average MAE of methods=========
    print("------1. Compute the average MAE of Methods------")
    aveMAE, gt2rs_mae = compute_ave_MAE_of_methods(gt_name_list,rs_dir_lists)
    print('\n')
    for i in range(0,len(rs_dirs)):
        print('>>%s: num_rs/num_gt-> %d/%d, aveMAE-> %.3f'%(rs_dirs[i], gt2rs_mae[i], len(gt_name_list), aveMAE[i]))

    ## 2. =======compute the Precision, Recall and F-measure of methods=========

    print('\n')
    print("------2. Compute the Precision, Recall and F-measure of Methods------")
    PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_name_list,rs_dir_lists,beta=0.3)
    for i in range(0,FM.shape[0]):
        print(">>", rs_dirs[i],":", "num_rs/num_gt-> %d/%d,"%(int(gt2rs_fm[i][0]),len(gt_name_list)), "maxF->%.3f, "%(np.max(FM,1)[i]), "meanF->%.3f, "%(np.mean(FM,1)[i]))
    print('\n')


    ## 3. =======Plot and save precision-recall curves=========
    print("------ 3. Plot and save precision-recall curves------")
    plot_save_pr_curves(PRE, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                        REC, # numpy array (num_rs_dir,255)
                        method_names = rs_dirs, # method names, shape (num_rs_dir), will be included in the figure legend
                        lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
                        linewidth = linewidth, # curve width, shape (num_rs_dir)
                        xrange = (0.5,1.0), # the showing range of x-axis
                        yrange = (0.5,1.0), # the showing range of y-axis
                        dataset_name = data_name, # dataset name will be drawn on the bottom center position
                        save_dir = data_dir, # figure save directory
                        save_fmt = 'png') # format of the to-be-saved figure
    print('\n')

    ## 4. =======Plot and save F-measure curves=========
    print("------ 4. Plot and save F-measure curves------")
    plot_save_fm_curves(FM, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                        mybins = np.arange(0,256),
                        method_names = rs_dirs, # method names, shape (num_rs_dir), will be included in the figure legend
                        lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
                        linewidth = linewidth, # curve width, shape (num_rs_dir)
                        xrange = (0.0,1.0), # the showing range of x-axis
                        yrange = (0.0,1.0), # the showing range of y-axis
                        dataset_name = data_name, # dataset name will be drawn on the bottom center position
                        save_dir = data_dir, # figure save directory
                        save_fmt = 'png') # format of the to-be-saved figure
    print('\n')

    print('Done!!!')

if __name__ == '__main__':
    gt_save_path = '/home/dmsheng/code/try/Evaluation-on-salient-object-detection/gt/coco'
    q_gt_save_path = '/home/dmsheng/code/try/Evaluation-on-salient-object-detection/pred/q_gt/coco'
    r_save_path = '/home/dmsheng/code/try/Evaluation-on-salient-object-detection/pred/re/coco'
    
    seg_gt_masks = np.ones((25, 128, 128))*255
    seg_q_gt_masks = np.ones((25, 128, 128))*255
    seg_q_gt_masks[:, 12:13, 24:25] = 0
    seg_re_masks = np.ones((25, 128, 128))*255
    seg_re_masks[:, 11:13, 24:27] = 0

    q_aveMAE, q_gt2rs_mae = compute_ave_MAE_of_methods(seg_gt_masks,seg_q_gt_masks)
    r_aveMAE, r_gt2rs_mae = compute_ave_MAE_of_methods(seg_gt_masks,seg_re_masks)
    print(f'q_num_rs/num_gt-> {int(q_gt2rs_mae)}/{len(seg_gt_masks)} q_MAE: {q_aveMAE}')
    print(f'r_num_rs/num_gt-> {int(r_gt2rs_mae)}/{len(seg_gt_masks)} r_MAE: {r_aveMAE}')

    q_PRE, q_REC, q_FM, q_gt2rs_fm = compute_PRE_REC_FM_of_methods(seg_gt_masks,seg_q_gt_masks,beta=0.3)
    r_PRE, r_REC, r_FM, r_gt2rs_fm = compute_PRE_REC_FM_of_methods(seg_gt_masks,seg_re_masks,beta=0.3)
    print(f'q_num_rs/num_gt-> {int(q_gt2rs_fm[0])}/{len(seg_gt_masks)}, q_maxF: {np.max(q_FM, 0)}, q_meanF: {np.mean(q_FM, 0)}')
    print(f'r_num_rs/num_gt-> {int(r_gt2rs_fm[0])}/{len(seg_gt_masks)}, q_maxF: {np.max(r_FM, 0)}, q_meanF: {np.mean(r_FM, 0)}')

    plot_save_pr_curves(np.stack([q_PRE, r_PRE]), # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                        np.stack([q_REC, r_REC]), # numpy array (num_rs_dir,255)
                        method_names = ['Quantize', 'Result'], # method names, shape (num_rs_dir), will be included in the figure legend
                        lineSylClr = ['r*', 'b*'], # curve styles, shape (num_rs_dir)
                        linewidth = [2, 2], # curve width, shape (num_rs_dir)
                        xrange = (0.5,1.0), # the showing range of x-axis
                        yrange = (0.5,1.0), # the showing range of y-axis
                        dataset_name = 'COCO', # dataset name will be drawn on the bottom center position
                        save_dir = './', # figure save directory
                        save_fmt = 'png') # format of the to-be-saved figure

    plot_save_fm_curves(np.stack([q_FM, r_FM]), # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                        mybins = np.arange(0,256),
                        method_names = ['Quantize', 'Result'], # method names, shape (num_rs_dir), will be included in the figure legend
                        lineSylClr = ['r-', 'b-'], # curve styles, shape (num_rs_dir)
                        linewidth = [2, 2], # curve width, shape (num_rs_dir)
                        xrange = (0.0,1.0), # the showing range of x-axis
                        yrange = (0.0,1.0), # the showing range of y-axis
                        dataset_name = 'COCO', # dataset name will be drawn on the bottom center position
                        save_dir = './', # figure save directory
                        save_fmt = 'png') # format of the to-be-saved figure