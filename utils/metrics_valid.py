
import numpy as np
import logging
import lpips
import torch
import cv2
import os
import matplotlib.pyplot as plt
from einops import rearrange

plt.switch_backend('agg')
np.seterr(divide='ignore', invalid='ignore')


def print_log(message, is_main_process=True):
    if is_main_process:
        print(message)
        logging.info(message)

def max_pool(arr, pool_size):

    arr = arr.squeeze(1)

    # 计算padding大小
    pad_H = pool_size - arr.shape[1] % pool_size
    pad_W = pool_size - arr.shape[2] % pool_size

    pad_size = ((0,0), (0,pad_H), (0,pad_W))
    
    # padding
    arr_padded = np.pad(arr, pad_size)
    # print(arr_padded)
    
    # 重新调整shape
    H = arr.shape[1] + pad_H
    W = arr.shape[2] + pad_W
    arr_reshaped = arr_padded.reshape(arr.shape[0], H//pool_size, pool_size,  W//pool_size, pool_size)
    arr_reshaped = arr_reshaped.transpose(0,1,3,2,4)

    # maxpool 
    arr_max_pooled = np.max(arr_reshaped, axis=(3,4))

    return arr_max_pooled

def cal_ssim(pred, true, data_range = 255):
    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def cal_cprs2(pred, true):
    '''cal cprs(continuous ranked probability score) in numpy, the data range is [0, 1]'''
    
    num_samples = pred.shape[0]
    absolute_error = np.mean(np.abs(pred - true), axis=0)
    
    # Calculate the ranks of the predicted probabilities and the observed outcomes
    pred_ranks = np.argsort(pred, axis=0)
    true_ranks = np.argsort(true, axis=0)
    
    # Calculate the differences between the ranks
    diff = pred_ranks - true_ranks
    
    # Calculate the weights for each rank
    weight = (np.arange(num_samples) + 1) / num_samples - 0.5
    
    # Calculate the CPRS score for each observation
    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    
    # Calculate the average CPRS score across all observations
    return np.average(per_obs_crps, weights=None)
    
    


class Evaluator(object):
    def __init__(self,  seq_len, value_scale, thresholds=[20, 30, 35, 40], **kwargs):
        self.metrics = {}
        self.thresholds = thresholds
        for threshold in self.thresholds:
            self.metrics[threshold] = {
                "hits": [],
                "misses": [],
                "falsealarms": [],
                "correctnegs": [],
                
                "hits44": [],
                "misses44": [],
                "falsealarms44": [],
                "correctnegs44": [],
                
                "hits16": [],
                "misses16": [],
                "falsealarms16": [],
                "correctnegs16": [],
            }
        self.seq_len = seq_len
        self.total = 0
        self.value_scale = value_scale 
    
    def float2int(self, arr):

        x = arr.clip(0.0, 1.0)
        x = x * self.value_scale
        x = x.astype(np.uint16)
        return x
        
    def evaluate(self, true_batch, pred_batch):
        # [batch_size, seq_len, channel height, wide], data_range [0.,1.]
        
        if isinstance(pred_batch, torch.Tensor):
            pred_batch = pred_batch.detach().cpu().numpy()
            true_batch = true_batch.detach().cpu().numpy()
        
        if not (true_batch.max() <= 1.0 and true_batch.min() >= 0.0):
            print_log(f"WARNING:: data max: {true_batch.max()}, min: {true_batch.min()}")
        
        pred_batch = pred_batch.clip(0.0, 1.0)
        true_batch = true_batch.clip(0.0, 1.0)
        
        assert pred_batch.shape == true_batch.shape, f"pred_batch.shape: {pred_batch.shape}, true_batch.shape: {true_batch.shape}"

        batch_size, seq_len = true_batch.shape[:2]
        
        pred = self.float2int(pred_batch)
        gt = self.float2int(true_batch)
        
        for threshold in self.thresholds:
            for b in range(batch_size):
                seq_hit, seq_miss, seq_falsealarm, seq_correctneg = [], [], [], []
                for t in range(seq_len):   
                    
                    hit, miss, falsealarm, correctneg = self.cal_frame(gt[b][t], pred[b][t], threshold)
                    seq_hit.append(hit)
                    seq_miss.append(miss)
                    seq_falsealarm.append(falsealarm)
                    seq_correctneg.append(correctneg)
                    
                self.metrics[threshold]["hits"].append(seq_hit)
                self.metrics[threshold]["misses"].append(seq_miss)
                self.metrics[threshold]["falsealarms"].append(seq_falsealarm)
                self.metrics[threshold]["correctnegs"].append(seq_correctneg)
            
                # 44
                hits44, misses44, falsealarms44, correctnegs44 = self.cal_frame(max_pool(gt[b], 4), max_pool(pred[b], 4), threshold)
                self.metrics[threshold]["hits44"].append(hits44)
                self.metrics[threshold]["misses44"].append(misses44)
                self.metrics[threshold]["falsealarms44"].append(falsealarms44)
                self.metrics[threshold]["correctnegs44"].append(correctnegs44)
                
                # 16
                hits16, misses16, falsealarms16, correctnegs16 = self.cal_frame(max_pool(gt[b], 16), max_pool(pred[b], 16), threshold)
                self.metrics[threshold]["hits16"].append(hits16)
                self.metrics[threshold]["misses16"].append(misses16)
                self.metrics[threshold]["falsealarms16"].append(falsealarms16)
                self.metrics[threshold]["correctnegs16"].append(correctnegs16)

        self.total += batch_size
            
        
    def cal_frame(self, obs, sim, threshold):
        obs = np.where(obs >= threshold, 1, 0)
        sim = np.where(sim >= threshold, 1, 0)

        # True positive (TP)
        hits = np.sum((obs == 1) & (sim == 1))

        # False negative (FN)
        misses = np.sum((obs == 1) & (sim == 0))

        # False positive (FP)
        falsealarms = np.sum((obs == 0) & (sim == 1))

        # True negative (TN)
        correctnegatives = np.sum((obs == 0) & (sim == 0))
        
        return hits, misses, falsealarms, correctnegatives
    
    
    def done(self, is_main_process=True):
        res_dict = {
            
        }
        print_log('*'*30+' < Evaluation Results: > '+'*'*30,)
        print_log(f"Total {self.total} samples with {self.seq_len} seq_len.")
        print_log('*'*90)
        
        avg_csi, avg_far, avg_pod, avg_hss = [], [], [], []
        avg_csi44, avg_csi16 = [], []
        for threshold in self.thresholds:
            hits = np.array(self.metrics[threshold]["hits"])
            misses = np.array(self.metrics[threshold]["misses"])
            falsealarms = np.array(self.metrics[threshold]["falsealarms"])
            correctnegs = np.array(self.metrics[threshold]["correctnegs"])
            
            # remove nan
            hits = np.nan_to_num(hits)
            misses = np.nan_to_num(misses)
            falsealarms = np.nan_to_num(falsealarms)
            correctnegs = np.nan_to_num(correctnegs)
            
            
            # first cal method

            csi1 = np.mean(hits, axis=0) / (np.mean(hits, axis=0) + np.mean(misses, axis=0) + np.mean(falsealarms, axis=0))
            far1 = np.mean(falsealarms, axis=0) / (np.mean(hits, axis=0) + np.mean(falsealarms, axis=0))
            pod1 = np.mean(hits, axis=0) / (np.mean(hits, axis=0) + np.mean(misses, axis=0))
            hss1 = 2 * (np.mean(hits, axis=0) * np.mean(correctnegs, axis=0) - np.mean(misses, axis=0) * np.mean(falsealarms, axis=0)) / ((np.mean(hits, axis=0) + np.mean(misses, axis=0)) * (np.mean(misses, axis=0) + np.mean(correctnegs, axis=0)) + (np.mean(hits, axis=0) + np.mean(falsealarms, axis=0)) * (np.mean(falsealarms, axis=0) + np.mean(correctnegs, axis=0)))                 
            
            csi1 = np.nan_to_num(csi1)
            far1 = np.nan_to_num(far1)
            pod1 = np.nan_to_num(pod1)
            hss1 = np.nan_to_num(hss1)
            
            avg_csi.append(np.mean(csi1))
            avg_far.append(np.mean(far1))
            avg_pod.append(np.mean(pod1))
            avg_hss.append(np.mean(hss1))
            
            hits44 = np.array(self.metrics[threshold]["hits44"])
            misses44 = np.array(self.metrics[threshold]["misses44"])
            falsealarms44 = np.array(self.metrics[threshold]["falsealarms44"])
            correctnegs44 = np.array(self.metrics[threshold]["correctnegs44"])
            
            hits16 = np.array(self.metrics[threshold]["hits16"])
            misses16 = np.array(self.metrics[threshold]["misses16"])
            falsealarms16 = np.array(self.metrics[threshold]["falsealarms16"])
            correctnegs16 = np.array(self.metrics[threshold]["correctnegs16"])
            
            # 4x4
            csi_pool44 = np.mean(hits44) / (np.mean(hits44) + np.mean(misses44) + np.mean(falsealarms44))
            avg_csi44.append(csi_pool44)
            
            # 16 x 16
            csi_pool16 = np.mean(hits16) / (np.mean(hits16) + np.mean(misses16) + np.mean(falsealarms16))
            avg_csi16.append(csi_pool16)
         
            # if threshold == 30:
            print_log('='*20 + f"Threshold: {threshold} with melthod 1"+'='*20)
            print_log(f'<CSI> : {np.mean(csi1)}; '+str(csi1))
            print_log(f'<FAR> : {np.mean(far1)}; '+str(far1))
            print_log(f'<POD> : {np.mean(pod1)}; '+str(pod1))
            print_log(f'<HSS> : {np.mean(hss1)}; '+str(hss1))
            
            print_log(f"< CSI_POOL 4x4 > : {csi_pool44}; CSI_POOL 16x16: {csi_pool16}")
            

        print_log('*'*20 + f"Overall Avg Metrics on Thresholds {self.thresholds}"+'*'*20)
        print_log(f"[ avg_csi ] : {np.mean(avg_csi)}; [ avg_far ] : {np.mean(avg_far)}; [ avg_pod ] : {np.mean(avg_pod)}; [ avg_hss] : {np.mean(avg_hss)}")
        print_log(f"[ avg_csi_pool 4x4 ] : {np.mean(avg_csi44)}; [ avg_csi_pool 16x16 ]: {np.mean(avg_csi16)}")

        
        res_dict['csi'] = np.nan_to_num(np.mean(avg_csi))
        print_log('='*90)
        
        return res_dict
    
    

    
if __name__ == '__main__':
    
    #  随机生成两个[]
    # from utils import read_samples, show_img_info
    # SCALE = 90.0
    
    # _, targets1, sampled1, mus1 = read_samples("/Users/demin/CodeSpace/MyWork/STD/exps/data_samples/100000-0-8")
    # _, targets2, sampled2, mus2 = read_samples("/Users/demin/CodeSpace/MyWork/STD/exps/data_samples/100000-0-9")
    
    
    
    # targets = np.array([targets1, targets2]) / SCALE
    # sampled = np.array([sampled1, sampled2]) / SCALE
    # mus = np.array([mus1, mus2]) / SCALE
    # # numpy 拓展维度
    
    # targets = np.expand_dims(targets, axis=2)
    # sampled = np.expand_dims(sampled, axis=2)
    # mus = np.expand_dims(mus, axis=2)
    
    # print(sampled.shape, targets.shape)
    # show_img_info(sampled)
    # show_img_info(targets)
    # show_img_info(mus)
     
    # evaluator = BatchEvaluator(value_scale=SCALE)
    # evaluator.evaluate(sampled, targets)
    # evaluator.done()
    
    # evaluator = BatchEvaluator(value_scale=SCALE)
    # evaluator.evaluate(mus, targets)
    # evaluator.done()
    
    tarray = np.array([i for i in range(36)]).reshape(1,6,6)
    # print(tarray.shape)
    print(tarray)
    pooled = max_pool(tarray, 2)
    tpooled = torch.max_pool2d(torch.from_numpy(tarray).float(), 2)

    print(pooled)
    print(tpooled)