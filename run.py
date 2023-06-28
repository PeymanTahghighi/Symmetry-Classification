from copy import deepcopy
from copyreg import pickle
import pickle
from random import uniform
from re import L
import cv2
import numpy as np
import os
from sklearn.model_selection import GridSearchCV,StratifiedKFold
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
import torch
from thorax import segment_thorax
from utility import get_symmetry_line, obscured_elipse
from tqdm import tqdm
from deep_learning.network import Unet
from deep_learning.model_trainer import NetworkTrainer
import config
from pystackreg import StackReg

def remove_blobs_spine(spine):
    kernel = np.ones((5,5), dtype=np.uint8);
    opening = cv2.morphologyEx(spine, cv2.MORPH_OPEN, kernel);
    ret_img = np.zeros_like(opening);
    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    all_area = [];
    for c in contours:
        area = cv2.contourArea(c);
        all_area.append(area);

    biggest = np.max(all_area);
    for idx, a in enumerate(all_area):
        if a > 0.4*biggest:
            ret_img = cv2.drawContours(ret_img,contours, idx, (255,255,255), -1);
    return ret_img;

def remove_blobs(ribs):
    kernel = np.ones((5,5), dtype=np.uint8);
    opening = cv2.morphologyEx(ribs, cv2.MORPH_OPEN, kernel);
    ret_img = np.zeros_like(opening);

    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    mean_area = 0;
    for c in contours:
        mean_area += cv2.contourArea(c);
    
    mean_area /= len(contours);
    
    for idx, p in enumerate(contours):
        area = cv2.contourArea(p);
        if area > mean_area * 0.1:
            ret_img = cv2.fillPoly(ret_img, [contours[idx]], (255,255,255));
            
    return ret_img;

def remove_outliers_hist_ver(hist, img):
    hist_thresh = np.where(hist.flatten() != 0)[0];
    streak_cnt = 0;
    streak_start = -1;
    streak_end = 0;
    min_streak = 1024*1024;
    min_start = 0;
    min_end = 0;
    streak_list = [];
    for idx in range(len(hist_thresh)-1):
        if hist_thresh[idx+1] - hist_thresh[idx] < 10:
            streak_cnt += 1;
            if streak_start == -1:
                streak_start = hist_thresh[idx];
        else:
            streak_end = hist_thresh[idx]+1 if hist_thresh[idx] < 1024 else hist_thresh[idx];
            streak_list.append([streak_start,streak_end,streak_end - streak_start]);
            if streak_cnt < min_streak:
                min_streak = streak_cnt;
                min_start = streak_start;
                min_end = streak_end;
            streak_start = -1;
            streak_end = -1;
            streak_cnt = 0;
    
    streak_end = hist_thresh[-1];
    streak_list.append([streak_start,streak_end,streak_end - streak_start]);
    streak_list.sort(key=lambda x:x[2],reverse=True);
    streak_list = np.array(streak_list);
    avg = np.mean(streak_list,axis=0)[2];
    img_new = deepcopy(img);
    for i in range(0,len(streak_list)):
        if streak_list[i][2] < avg*0.65:
            img_new[:,streak_list[i][0]:streak_list[i][1]] = 0
    return img_new;

def draw_missing_spine(img):
    img_cpy = deepcopy(img);
    rows = np.sum(img, axis = 1);
    rows_thresh = np.where(rows > 0)[0];
    GROW = 10;
    if rows_thresh[0] != 0:
        avg_w = [];
        for r in range(int(len(rows_thresh)*0.1)):
            cols_thresh = np.where(img[rows_thresh[r],:]>0)[0];
            w = cols_thresh[-1] - cols_thresh[0];
            avg_w.append(w);
        avg_w = np.mean(avg_w);

        for r in rows_thresh:
            cols_thresh = np.where(img[r,:]>0)[0];
            w = cols_thresh[-1] - cols_thresh[0];
            if w > avg_w*0.9:
                img_cpy[:r,cols_thresh[0]-GROW:cols_thresh[-1]+GROW] = 255;
                break;
    if rows_thresh[-1] != 1024:
        rows_thresh = rows_thresh[::-1];
        avg_w = [];
        for r in range(int(len(rows_thresh)*0.1)):
            cols_thresh = np.where(img[rows_thresh[r],:]>0)[0];
            w = cols_thresh[-1] - cols_thresh[0];
            avg_w.append(w);
        avg_w = np.mean(avg_w);
        for r in rows_thresh:
            cols_thresh = np.where(img[r,:]>0)[0];
            w = cols_thresh[-1] - cols_thresh[0];
            if w > avg_w*0.9:
                img_cpy[r:1024,cols_thresh[0]-GROW:cols_thresh[-1]+GROW] = 255;
                break;
    
    return img_cpy;

def crop_top(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    for c in contours:
        if cv2.contourArea(c) > 10:
            ch = cv2.convexHull(c);
            #cv2.drawContours(img, [ch], -1, (255,255,255), 2);
            extLeft = tuple(ch[ch[:, :, 0].argmin()][0])
            extRight = tuple(ch[ch[:, :, 0].argmax()][0])
            extTop = tuple(ch[ch[:, :, 1].argmin()][0])
            extBot = tuple(ch[ch[:, :, 1].argmax()][0])
            total_height = np.abs(extTop[1] - extBot[1]);
            height_to_crop = int(total_height*0.25) + extTop[1]
            perimeter = cv2.arcLength(ch, True);
    
    return img[:height_to_crop,:];

def get_perimeter(img):
    img_crop = crop_top(img);
    contours_crop, _ = cv2.findContours(img_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    #assert len(contours) == 1, "Number of contours detected should be exactly one";
    #hull_list = [];
    perimeter = 0;
    perimeter_crop = 0;
    for c in contours_crop:
        if cv2.contourArea(c) > 10:
            ch = cv2.convexHull(c);
            #hull_list.append(ch);
            
            perimeter_crop = cv2.arcLength(ch, True); 
    
    for c in contours:
        if cv2.contourArea(c) > 10:
            ch = cv2.convexHull(c);
            #hull_list.append(ch);
            
            perimeter = cv2.arcLength(ch, True); 
    
    return perimeter, perimeter_crop;

def get_histogram(img, bins):
    temp_img = np.where(img == 255, 1, 0);
    h,w = img.shape;
    if h < bins:
        ph = bins;
        padded_img = np.zeros((ph,w));
        padded_img[:h,:] = img;
        img = padded_img;
        h = ph;

    rows_per_bin = int(h / bins);
    hist_horizontal = [];
    for i in range(0,h,rows_per_bin):
        s = temp_img[i:i+rows_per_bin,:];
        hist_horizontal.append(int(s.sum()));
    
    hist_horizontal = np.array(hist_horizontal, dtype=np.float32);
    hist_horizontal = np.expand_dims(hist_horizontal, axis=1);
    hist_horizontal = hist_horizontal / hist_horizontal.sum();

    hist_vertical = [];
    for i in range(0,w,rows_per_bin):
        s = temp_img[:,i:i+rows_per_bin];
        hist_vertical.append(int(s.sum()));
    
    hist_vertical = np.array(hist_vertical, dtype=np.float32);
    hist_vertical = np.expand_dims(hist_vertical, axis=1);
    hist_vertical = hist_vertical / hist_vertical.sum();
    
    return hist_horizontal, hist_vertical;

def IoU(img_1, img_2):
    h1,w = img_1.shape;
    h2,_ = img_2.shape;

    h = max(h1, h2);

    img_1_tmp = cv2.resize(img_1, (w,h));
    img_2_tmp = cv2.resize(img_2, (w,h));

    img_img_2_flipped = cv2.flip(img_2_tmp, 1);
    sr = StackReg(StackReg.TRANSLATION); 
    sr.register(img_1_tmp, img_img_2_flipped);
    out = sr.transform(img_img_2_flipped);
    out = np.array(out, dtype=np.uint8);
    intersection = cv2.bitwise_and(out, img_1_tmp);
    intersection = np.where(intersection == 255, 1, 0);
    intersection = np.sum(intersection);
    union = cv2.bitwise_or(out, img_1_tmp);
    union = np.where(union == 255, 1, 0).sum();
    xor = cv2.bitwise_xor(out, img_1_tmp)
    xor = (np.where(xor == 255, 1, 0).sum()) / (w*h);
    iou = intersection / union;
    return iou, out, xor;

def cross_entropy(p,q):
    return np.sum(-p*np.log(q));

def JSD(p,q):
    p = p + config.EPSILON;
    q = q + config.EPSILON;
    avg = (p+q)/2;
    jsd = (cross_entropy(p,avg) - cross_entropy(p,p))/2 + (cross_entropy(q,avg) - cross_entropy(q,q))/2;
    #clamp
    if jsd > 1.0:
        jsd = 1.0;
    elif jsd < 0.0:
        jsd = 0.0;
    
    return jsd;

def get_corner(mask, rev = False):
    w,h = mask.shape;
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    max_area = 0;
    max_cnt = None;
    for c in contours:
        a = cv2.contourArea(c);
        if a > max_area:
            max_area = a;
            max_cnt = c;
    max_cnt = np.squeeze(max_cnt);
    if rev is True:
        max_cnt[:,0]= h - max_cnt[:,0];
        # tmp = np.zeros_like(mask);
        # tmp = cv2.drawContours(tmp, [max_cnt], 0, (255,255,255), -1);
        # cv2.imshow('orig', mask);
        # cv2.imshow('rev', tmp);
        # cv2.waitKey();
        
        
    max_cnt = cv2.approxPolyDP(max_cnt, cv2.arcLength(max_cnt, True)*0.01, True);
    mask_ret = np.zeros_like(mask);
    mask_ret = cv2.drawContours(mask_ret, [max_cnt], 0, (255,255,255), -1);
    pints = max_cnt.squeeze();
    top = pints[np.argmin(pints[:,1])];
    s = np.sum(pints, axis = 1);
    crn_top_left = pints[np.argmin(s)];
    a = np.abs(pints[:,0]-pints[:,1]);
    crn_top_right = pints[np.argmin((pints[:,1]-pints[:,0]))];

    diff = np.sqrt((top[0]-crn_top_right[0])**2 + (top[1]-crn_top_right[1])**2);
    mask_ret = cv2.cvtColor(mask_ret, cv2.COLOR_GRAY2RGB);
    
    if diff < 0.1:
        if rev is False:
            mask_ret = cv2.circle(mask_ret, (crn_top_left[0], crn_top_left[1]), 10, (255,0,0), -1);
        ret_point = crn_top_left;
    else:
        if rev is False:
            mask_ret = cv2.circle(mask_ret, (top[0], top[1]), 2, (0,0,255), -1);
        ret_point = top;
    
    if rev is True:
        ret_point[0] = h - ret_point[0];
        mask_ret = cv2.circle(mask_ret, (ret_point[0], ret_point[1]), 5, (255,0,0,), -1);
    return ret_point, mask_ret;

def centroid(img):
    M = cv2.moments(img);
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY;

def extract_symmetry_features(img_left, img_right, spine):
    img_crop_left = crop_top(img_left);
    img_crop_right = crop_top(img_right);
    w,h = img_left.shape;
    area_left = (img_left == 255).sum() / (w*h);
    area_left_crop = (img_crop_left == 255).sum() / (img_crop_left.shape[0]*img_crop_left.shape[1]);
    peri_left, peri_left_crop = get_perimeter(img_left);

    area_right = (img_right == 255).sum() / (w*h);
    area_right_crop = (img_crop_right == 255).sum() / (img_crop_right.shape[0]*img_crop_right.shape[1]);
    peri_right, peri_right_crop = get_perimeter(img_right);


    f = min(area_left, area_right) / max(area_right, area_left);
    f_crop = min(area_left_crop, area_right_crop) / max(area_right_crop, area_left_crop);

    s = 0;
    s = min(peri_left, peri_right) / max(peri_right,peri_left);
    s_crop = min(peri_left_crop, peri_right_crop) / max(peri_right_crop, peri_left_crop);

    max_h = max(img_crop_left.shape[0], img_crop_right.shape[0]);
    img_crop_right = cv2.resize(img_crop_right, (512, max_h));
    img_crop_left = cv2.resize(img_crop_left, (512, max_h));


    hist_left_hor, hist_left_ver = get_histogram(img_left, 256);
    hist_left_crop_hor, hist_left_crop_ver = get_histogram(img_crop_left, 256);
    hist_right_hor, _ = get_histogram(img_right, 256);
    hist_right_crop_hor, hist_right_crop_ver = get_histogram(img_crop_right, 256);
    iou,img_right_flipped, xor = IoU(img_left, img_right);
    _, hist_right_ver = get_histogram(img_right_flipped, 256);

    jsd1 = JSD(hist_left_hor, hist_right_hor);
    jsd2 = JSD(hist_left_ver, hist_right_ver);
    jsd3 = JSD(hist_left_crop_hor, hist_right_crop_hor);
    jsd4 = JSD(hist_left_crop_ver, hist_right_crop_ver);
    diff1 = cv2.compareHist(hist_left_hor, hist_right_hor, cv2.HISTCMP_INTERSECT);
    diff2 = cv2.compareHist(hist_left_hor, hist_right_hor, cv2.HISTCMP_BHATTACHARYYA);
    diff3 = cv2.compareHist(hist_left_hor, hist_right_hor, cv2.HISTCMP_CHISQR);
    diff4 = cv2.compareHist(hist_left_hor, hist_right_hor, cv2.HISTCMP_CORREL);
    iou_crop_lr, out1, xor_crop_lr = IoU(img_crop_left, img_crop_right);
    iou_crop_rl, out2, xor_crop_rl = IoU(img_crop_right,img_crop_left);

    iou_lr, out1, xor_lr = IoU(img_left, img_right);
    iou_rl, out2, xor_rl = IoU(img_right,img_left);
    
    sym_line = get_symmetry_line(spine);

    top_left, ret_mask_left = get_corner(img_left);
    top_right, ret_mask_right = get_corner(img_right, True);

    dist_left = 0;
    x_left = top_left[0];
    x_right = top_right[0];
    x_avg_left = np.mean(sym_line[max(top_left[1]-10,0):top_left[1]+10,1]);
    x_avg_right = np.mean(sym_line[max(top_right[1]-10,0):top_right[1]+10,1]);
    dist_left = np.abs(x_left - x_avg_left);
    dist_right = np.abs(x_right - x_avg_right);
    ratio = min(dist_left, dist_right) / max(dist_left, dist_right);

    cX_l, cY_l = centroid(img_left);
    cX_r, cY_r = centroid(img_right);

    cX_f_l = sym_line[cX_l,1];

    cX_f_r = sym_line[cX_r,1];

    left_to_center = abs(cX_f_l - cX_l);
    right_to_center = abs(cX_f_r - cX_r);
    ratio_centroid = min(left_to_center, right_to_center) / max(left_to_center, right_to_center);

    feat = [];
    feat.append(f);
    feat.append(s);
    feat.append(s_crop);
    feat.append(f_crop);
    feat.append(jsd1);
    feat.append(jsd2);
    feat.append(jsd3);
    feat.append(jsd4);
    feat.append(diff1);
    feat.append(diff2);
    feat.append(diff3);
    feat.append(diff4);
    feat.append(iou_crop_lr);
    feat.append(iou_crop_rl);
    feat.append(xor_crop_lr);
    feat.append(xor_crop_rl);
    feat.append(iou_lr);
    feat.append(iou_rl);
    feat.append(xor_lr);
    feat.append(xor_rl);
    feat.append(ratio);
    feat.append(ratio_centroid);


    return feat;

def remove_outliers_hist_hor(hist, img):
    hist_thresh = np.where(hist.flatten() != 0)[0];
    streak_cnt = 0;
    streak_start = -1;
    streak_end = 0;
    max_streak = 0;
    max_start = 0;
    max_end = 0;
    for idx in range(len(hist_thresh)-1):
        if hist_thresh[idx+1] - hist_thresh[idx] < 50:
            streak_cnt += 1;
            if streak_start == -1:
                streak_start = hist_thresh[idx];
        else:
            streak_end = hist_thresh[idx];
            if streak_cnt > max_streak:
                max_streak = streak_cnt;
                max_start = streak_start;
                max_end = streak_end;
            streak_start = -1;
            streak_end = -1;
            streak_cnt = 0;
    
    streak_end = hist_thresh[idx+1];
    if streak_cnt > max_streak:
        max_streak = streak_cnt;
        max_start = streak_start;
        max_end = streak_end;
    img_new = deepcopy(img);
    img_new[:max_start,:] = 0
    img_new[max_end+1:,:] = 0

    return img_new;


def build_thorax():
    spine_and_ribs_segmentation_model = Unet(1).to(config.DEVICE);
    for idx in range(5):
        spine_and_ribs_segmentation_model.load_state_dict(pickle.load(open(f'results\\hemi\\spine and ribs_{idx}.pt', 'rb')));

        f = pickle.load(open(f'{idx}.dump', 'rb'));
        train_x, train_mask, train_lbl,_, test_x, test_mask, test_lbl,test_exp_lbl = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7];

        #train
        for i in tqdm(range(len(train_x))):
            
            file_name = os.path.basename(train_x[i]);
            file_name = file_name[:file_name.rfind('.')];
            ndex = img_list.index(file_name);
            lbl = sym_list[ndex];
            if lbl == 1:
                lbl = 0;
            elif lbl == 2:
                lbl = 1;

            total_Y.append(lbl);
            if file_name not in feat_dict.keys():
                left = cv2.imread(f'results\\train_data\\{file_name}_left.png', cv2.IMREAD_GRAYSCALE);
                right = cv2.imread(f'results\\train_data\\{file_name}_right.png', cv2.IMREAD_GRAYSCALE);
                spine_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{file_name}.meta', 'rb'));
                spine_name = spine_meta['Spine'][2];
                spine_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{spine_name}', cv2.IMREAD_GRAYSCALE);
                spine_mask = np.where(spine_mask>0, 255, 0);
                spine_mask = cv2.resize(spine_mask.astype("uint8"), (1024,1024));
                spine_mask = draw_missing_spine(spine_mask);
                

                feat = extract_symmetry_features(left, right, spine_mask)
                feat_dict[file_name] = feat;
            else:
                feat = feat_dict[file_name];

            total_X.append(feat);


        # #--------------------------------------------------------------------------------------------------------

        #test
        for i in tqdm(range(len(test_x))):
            file_name = os.path.basename(test_x[i]);
            file_name = file_name[:file_name.rfind('.')];
            if os.path.exists(f'results_obscured_thorax\\{idx}\\{file_name}_left.png') is False:
                radiograph_image = cv2.imread(f"obscured_dataset\\{file_name}.png",cv2.IMREAD_GRAYSCALE);

                mask_full = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Full radiograph segmentation\\labels\\', f'{file_name}_0.png'), cv2.IMREAD_GRAYSCALE);
                mask_full = cv2.resize(mask_full, (radiograph_image.shape[1], radiograph_image.shape[0]));
                mask_full = np.where(mask_full > 0, 1, 0);

                radiograph_image = (radiograph_image*mask_full).astype("uint8");
                #cv2.imwrite(f'test\\{file_name}.png', radiograph_image);
                cv2.imwrite(f'results_obscured_thorax\\{idx}\\{file_name}_rad.png', radiograph_image);
                radiograph_image = np.repeat(np.expand_dims(radiograph_image, axis=2), 3,axis=2);


                transformed = config.valid_transforms(image = radiograph_image);
                radiograph_image = transformed["image"];
                radiograph_image = radiograph_image.to(config.DEVICE);
                
                #spine and ribs
                out = spine_and_ribs_segmentation_model(radiograph_image.unsqueeze(dim=0));
                out = torch.sigmoid(out).permute(0,2,3,1) > 0.5;
                out = out.detach().cpu().numpy().squeeze()*255;

                # out = (torch.softmax(out, dim= 1)[0].permute(1,2,0)).detach().cpu().numpy();
                # out = np.argmax(out,axis = 2);

                # ribs = (out == 1).astype("uint8")*255;
                # spine = (out == 2).astype("uint8")*255;

                # ribs_new = remove_blobs(ribs);
                # spine_new = remove_blobs_spine(spine).astype("uint8");
                # spine_new = draw_missing_spine(spine_new);

                # hist_hor, hist_ver = get_histogram(ribs_new,1024);
                # ribs_new_out = remove_outliers_hist_ver(hist_ver, ribs_new);
                # ribs_new_out = remove_outliers_hist_hor(hist_hor, ribs_new_out);

                # sym_line = get_symmetry_line(spine_new); 
                # ribs_left, ribs_right = divide_image_symmetry_line(ribs_new_out, sym_line);
                # thorax_left = segment_thorax(ribs_left);
                # thorax_right = segment_thorax(ribs_right);
                # #whole_thorax = segment_thorax(ribs_new_out);

                # cv2.imwrite(f'results_obscured\\{idx}\\{file_name}_left.png', thorax_left);
                # cv2.imwrite(f'results_obscured\\{idx}\\{file_name}_right.png', thorax_right);
                cv2.imwrite(f'results_obscured_thorax\\{idx}\\{file_name}.png', out.astype("uint8"));
                # cv2.imwrite(f'results_obscured\\{idx}\\{file_name}_ribs.png', ribs_new_out);

def build_experiment_dataset(type):
    for idx in range(5):
        f = pickle.load(open(f'{idx}.dump', 'rb'));
        train_x, train_mask, train_lbl,_, test_x, test_mask, test_lbl,test_exp_lbl = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7];

        #test
        for i in tqdm(range(len(test_x))):
            file_name = os.path.basename(test_x[i]);
            file_name = file_name[:file_name.rfind('.')];
            #if os.path.exists(f'results_thorax\\{idx}\\{file_name}.png') is False:
            radiograph_image = cv2.imread(test_x[i],cv2.IMREAD_GRAYSCALE);

            exp_lbl = test_exp_lbl[i];
            mask_full = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Full radiograph segmentation\\labels\\', f'{file_name}_0.png'), cv2.IMREAD_GRAYSCALE);
            mask_full = cv2.resize(mask_full, (radiograph_image.shape[1], radiograph_image.shape[0]));
            mask_full = np.where(mask_full > 0, 1, 0);
            if type == 'overexposure':
                if exp_lbl == '0.0':
                    radiograph_image_trans = config.overexposure_transforms(image = radiograph_image);
                    radiograph_image = radiograph_image_trans["image"];
            elif  type == 'underexposure':
                 if exp_lbl == '0.0':
                    radiograph_image_trans = config.underexposure_transforms(image = radiograph_image);
                    radiograph_image = radiograph_image_trans["image"];
            elif type == 'obscured':
                left = cv2.imread(f'results\\train_data\\{file_name}_left.png', cv2.IMREAD_GRAYSCALE);
                right = cv2.imread(f'results\\train_data\\{file_name}_right.png', cv2.IMREAD_GRAYSCALE);
                radiograph_image = obscured_elipse(radiograph_image,left,right);
               
            cv2.imwrite(f'obscured_dataset\\{file_name}.png', radiograph_image.astype("uint8"));


def compare_thorax_segmentation():
    all_iou = [];
    for idx in range(5):
        all_iou_fold = [];
        f = pickle.load(open(f'{idx}.dump', 'rb'));
        train_x, train_mask, train_lbl,_, test_x, test_mask, test_lbl,_ = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7];
        for t in test_x:
            file_name = os.path.basename(t);
            file_name = file_name[:file_name.rfind('.')];
            pred = cv2.imread(f'results_obscured_thorax\\{idx}\\{file_name}.png', cv2.IMREAD_GRAYSCALE);
            pred = np.where(pred>0, 1, 0);
            gt_left = cv2.imread(f'results\\train_data\\{file_name}_left.png', cv2.IMREAD_GRAYSCALE);
            gt_right = cv2.imread(f'results\\train_data\\{file_name}_right.png', cv2.IMREAD_GRAYSCALE);
            gt = gt_left + gt_right;
            gt = cv2.resize(gt, (768,768))
            gt = np.where(gt>0, 1, 0);

            intersec = np.sum(gt*pred);
            union = np.sum(np.logical_or(gt,pred));
            iou = intersec / union;
            all_iou_fold.append(iou);
        
        print(f'fold: {idx}: {np.mean(all_iou_fold)}');
        all_iou.append(np.mean(all_iou_fold));
    print(f'result: {np.mean(all_iou)}');

def optimize_symmetry_model():
    total_X = [];
    total_Y = [];
    train_fold_indices = [];
    test_fold_indices = [];
    df = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    sym_list = list(df['Symmetric Hemithoraces']);
    img_list = list(map(str, list(df['Image'])));

    for i in tqdm(range(len(img_list))):
        file_name = img_list[i];
        if os.path.exists(f'results\\train_data\\{file_name}.png') is False:
            #if os.path.exists(f'results\\train_data\\{file_name}.png') is False:
            # lbl = sym_list[img_list.index(train_imgs[i])];
            # if lbl ==0 or lbl == 1:
            #     lbl = 0;
            # else:
            #     lbl = 1;
            # total_Y.append(lbl);
            meta_data = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{file_name}.meta', 'rb'));
                
            spine_mask = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs', 'labels', meta_data['Spine'][2]), cv2.IMREAD_GRAYSCALE);
            #spine_mask = np.where(spine_mask > 0, 1, 0).astype("uint8");
            ribs_mask = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs', 'labels', meta_data['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
            #ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.uint8);

            whole_thorax = segment_thorax(ribs_mask);
            # spine_mask = cv2.resize(spine_mask, (1024,1024));
            # sym_line = get_symmetry_line(spine_mask);
            # ribs_left, ribs_right = divide_image_symmetry_line(whole_thorax, sym_line);
            # thorax_left = segment_thorax(ribs_left);
            #thorax_right = segment_thorax(ribs_right);

            cv2.imwrite(f'results\\train_data\\{file_name}.png', whole_thorax);
            #cv2.imwrite(f'results\\train_data\\{file_name}_right_thorax.png', ribs_right);
            #cv2.imwrite(f'results\\train_data\\{file_name}_left.png', thorax_left);
            #cv2.imwrite(f'results\\train_data\\{file_name}_right.png', thorax_right);

    
    spine_and_ribs_segmentation_model = Unet(3).to(config.DEVICE);

    # #data = pickle.load(open('data.dmp', 'rb'));
    # #total_x, _, custom_cv = data[0], data[1], data[2];

    feat_dict = dict();

    # for idx in range(5):
    #     spine_and_ribs_segmentation_model.load_state_dict(pickle.load(open(f'results\\U-Net\\spine and ribs_{idx}.pt', 'rb')));

    #     f = pickle.load(open(f'{idx}.dump', 'rb'));
    #     train_x, train_mask, train_lbl, _, test_x, test_mask, test_lbl, _ = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7];

    #     train_fold_indices.append(np.arange(len(total_X),len(total_X)+len(train_x)));
    #     test_fold_indices.append(np.arange(len(total_X)+len(train_x),len(total_X)+len(train_x)+len(test_x)));

    # #     #train
        
    #     for i in tqdm(range(len(train_x))):
            
    #         file_name = os.path.basename(train_x[i]);
    #         file_name = file_name[:file_name.rfind('.')];
    #         ndex = img_list.index(file_name);
    #         lbl = sym_list[ndex];
    #         if lbl == 1:
    #             lbl = 0;
    #         elif lbl == 2:
    #             lbl = 1;

    #         total_Y.append(lbl);
    #         if file_name not in feat_dict.keys():
    #             left = cv2.imread(f'results\\train_data\\{file_name}_left_thorax.png', cv2.IMREAD_GRAYSCALE);
    #             right = cv2.imread(f'results\\train_data\\{file_name}_right_thorax.png', cv2.IMREAD_GRAYSCALE);
    #             spine_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{file_name}.meta', 'rb'));
    #             spine_name = spine_meta['Spine'][2];
    #             spine_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{spine_name}', cv2.IMREAD_GRAYSCALE);
    #             spine_mask = np.where(spine_mask>0, 255, 0);
    #             spine_mask = cv2.resize(spine_mask.astype("uint8"), (1024,1024));
    #             spine_mask = draw_missing_spine(spine_mask);
                
    #             feat = extract_symmetry_features(left, right, spine_mask)
    #             feat_dict[file_name] = feat;
    #         else:
    #             feat = feat_dict[file_name];

    #         total_X.append(feat);


    #     # #--------------------------------------------------------------------------------------------------------

    #     # total_X = np.array(total_X);
    #     # #total_Y = np.expand_dims(np.array(total_Y),axis = 1);
    #     # koft = np.zeros((total_X.shape[0], total_X.shape[1]+1));
    #     # koft[:,:total_X.shape[1]] = total_X;
    #     # koft[:,20] = total_Y;
    #     # df = pd.DataFrame(koft, columns=[np.arange(0,21)]);
    #     # sns.heatmap(df.corr(), annot=True);
    #     # plt.show();

    #     #test
    #     for i in tqdm(range(len(test_x))):
    #         file_name = os.path.basename(test_x[i]);
    #         file_name = file_name[:file_name.rfind('.')];
    #         radiograph_image = cv2.imread(test_x[i],cv2.IMREAD_GRAYSCALE);
    #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #         radiograph_image = clahe.apply(radiograph_image);
    #         radiograph_image = np.expand_dims(radiograph_image, axis=2);
    #         radiograph_image = np.repeat(radiograph_image, 3,axis=2);


    #         transformed = config.valid_transforms(image = radiograph_image);
    #         radiograph_image = transformed["image"];
    #         radiograph_image = radiograph_image.to(config.DEVICE);
            
    #         #spine and ribs
    #         out = spine_and_ribs_segmentation_model(radiograph_image.unsqueeze(dim=0));
    #         out = (torch.softmax(out, dim= 1)[0].permute(1,2,0)).detach().cpu().numpy();
    #         out = np.argmax(out,axis = 2);

    #         ribs = (out == 1).astype("uint8")*255;
    #         spine = (out == 2).astype("uint8")*255;

    #         #ribs_new = remove_blobs(ribs);
    #         spine_new = remove_blobs_spine(spine).astype("uint8");
    #         spine_new = draw_missing_spine(spine_new);

    #         whole_thorax = cv2.imread(f'results\\{idx}\\{file_name}.png', cv2.IMREAD_GRAYSCALE);
    #         sym_line = get_symmetry_line(spine_new);
    #         ribs_left, ribs_right = divide_image_symmetry_line(whole_thorax, sym_line);
    #         #thorax_left = segment_thorax(ribs_left);
    #         #thorax_right = segment_thorax(ribs_right);

    #         cv2.imwrite(f'results\\{idx}\\{file_name}_left_thorax.png', ribs_left);
    #         cv2.imwrite(f'results\\{idx}\\{file_name}_right_thorax.png', ribs_right);

    #         left = cv2.imread(f'results\\{idx}\\{file_name}_left_thorax.png', cv2.IMREAD_GRAYSCALE);
    #         right = cv2.imread(f'results\\{idx}\\{file_name}_right_thorax.png', cv2.IMREAD_GRAYSCALE);
    #         feat = extract_symmetry_features(left, right, spine_new);
    #         total_X.append(feat);
    #         ndex = img_list.index(file_name);
    #         lbl = sym_list[ndex];
    #         if lbl == 1:
    #             lbl = 0;
    #         elif lbl == 2:
    #             lbl = 1;

    #         total_Y.append(lbl);

    #         # hist_hor, hist_ver = get_histogram(ribs_new,1024);
    #         # ribs_new_out = remove_outliers_hist_ver(hist_ver, ribs_new);
    #         # ribs_new_out = remove_outliers_hist_hor(hist_hor, ribs_new_out);
    #         # f = int(np.sum(ribs_new_out));
    #         # n = int(np.sum(ribs_new));
    #         # a = int(np.sum(ribs_new_out)) == int(np.sum(ribs_new));
    #         # if a is False:
    #         #     print(file_name);

    #         #     sym_line = get_symmetry_line(spine_new);
    #         #     ribs_left, ribs_right = divide_image_symmetry_line(ribs_new_out, sym_line);
    #         #     thorax_left = segment_thorax(ribs_left);
    #         #     thorax_right = segment_thorax(ribs_right);

    #         #     cv2.imwrite(f'results\\{idx}\\{file_name}_left.png', thorax_left);
    #         #     cv2.imwrite(f'results\\{idx}\\{file_name}_right.png', thorax_right);

    #         # left = cv2.imread(f'results\\{idx}\\outputs\\{t}_left.png', cv2.IMREAD_GRAYSCALE);
    #         # right = cv2.imread(f'results\\{idx}\\outputs\\{t}_right.png', cv2.IMREAD_GRAYSCALE);
    #         # spine = cv2.imread(f'results\\{idx}\\outputs\\{t}_spine.png', cv2.IMREAD_GRAYSCALE);
    #         # full = cv2.imread(f'results\\{idx}\\outputs\\{t}_thorax.png', cv2.IMREAD_GRAYSCALE);
    #         # symmetry_features = extract_symmetry_features(left, right, spine, full);
    #         # total_X.append(symmetry_features);
    #     #----------------------------------------------------------------------------------------

    # custom_cv = zip(train_fold_indices, test_fold_indices);
    # pickle.dump([total_X, total_Y, custom_cv], open('data_thorax.dmp', 'wb'));
    # return;
        
    data = pickle.load(open('data_thorax.dmp', 'rb'));
    total_x, total_y, custom_cv = data[0], data[1], data[2];

    # 
    # total_y = np.array(total_y);
    # total_x = np.array(total_x);
    # for train_id, test_id in custom_cv:
    #     train_x = total_x[train_id];
    #     train_y = total_y[train_id];
    #     koft = np.zeros((train_x.shape[0], train_x.shape[1]+1));
    #     koft[:,:train_x.shape[1]] = train_x;
    #     koft[:,22] = np.array(train_y, np.int32);
    #     df = pd.DataFrame(koft, columns=[np.arange(0,23)]);
    #     sns.heatmap(df.corr(), annot=True);
    #     plt.show();

    # total_X = np.array(total_x);
    # total_Y = np.expand_dims(np.array(total_Y),axis = 1);
    # koft = np.zeros((total_X.shape[0], total_X.shape[1]+1));
    # koft[:,:total_X.shape[1]] = total_X;
    # koft[:,22] = np.array(total_y, np.int32);
    # df = pd.DataFrame(koft, columns=[np.arange(0,23)]);
    # sns.heatmap(df.corr(), annot=True);
    # plt.show();



    total_x = np.array(total_x);
    total_y = np.array(total_y, np.int32);

    #Best idx: 2772  best score: 0.7813729298273577  best comb: (0, 2, 3, 4, 6, 20, 13)
    #best_param: {'gbc__learning_rate': 0.1, 'gbc__max_depth': 3, 'gbc__n_estimators': 100}

    # Best idx: 3347  best score: 0.8095606555922161  best comb: (0, 1, 2, 3, 6, 9, 20, 21)
    # best_param: {'mlp__activation': 'tanh', 'mlp__alpha': 0.1, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'invscaling', 'mlp__solver': 'adam'}
    
    # feature_set_mlp = [0, 1, 2, 3, 6, 9, 20, 21];
    # feature_set_gbc = [0, 2, 3, 4, 6, 20, 13];
    # feature_set_svc = [0, 3, 20];

    # features_mlp = np.zeros((total_x.shape[0], len(feature_set_mlp)));
    # for j in range(len(feature_set_mlp)):
    #     features_mlp[:,j] = total_x[:,feature_set_mlp[j]];
    
    # features_gbc = np.zeros((total_x.shape[0], len(feature_set_gbc)));
    # for j in range(len(feature_set_gbc)):
    #     features_gbc[:,j] = total_x[:,feature_set_gbc[j]];
    
    # features_svc = np.zeros((total_x.shape[0], len(feature_set_svc)));
    # for j in range(len(feature_set_svc)):
    #     features_svc[:,j] = total_x[:,feature_set_svc[j]];

    # total_f1 = [];
    # for train_id, test_id in custom_cv:
    #     train_y, test_y = total_y[train_id],total_y[test_id];
    #     clf1 = Pipeline([('scalar',RobustScaler()),('mlp',MLPClassifier(50,"tanh", solver = 'adam', learning_rate='invscaling', alpha = 0.1, max_iter=500))]);
    #     clf2 = Pipeline([('scalar',RobustScaler()), ('gbc', GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=100))]);
    #     clf3 = Pipeline([('scalar',RobustScaler()),('svc', SVC(C=0.1, gamma=0.01, kernel='rbf'))]);

    #     clf1.fit(features_mlp[train_id], train_y);
    #     clf2.fit(features_gbc[train_id], train_y);
    #     clf3.fit(features_svc[train_id], train_y);

    #     pred1 = clf1.predict(features_mlp[test_id]);
    #     pred2 = clf2.predict(features_gbc[test_id]);
    #     pred3 = clf3.predict(features_svc[test_id]);

    #     maj = pred1 + pred2 + pred3;
    #     maj = np.where(maj > 1, 1, 0);
    #     prec, rec, f1, _ = precision_recall_fscore_support(test_y, pred3, average='binary');
    #     r = roc_auc_score(test_y, pred3)
        
    #     total_f1.append(r);
    
    # print(np.mean(total_f1));

    #lbl_list = np.expand_dims(lbl_list, axis = 1);
    best_idx = -1;
    best_score = 0;
    ps = list(powerset([0,1,2,3,4,6,8,20,21,13,12, 16,17]));
    for i in tqdm(range(0, len(ps))):
        if len(ps[i]) > 1:
            features = np.zeros((total_x.shape[0], len(ps[i])),dtype=np.float32);
            for idx, p in enumerate(ps[i]):
                features[:,idx] = total_x[:,p];
            
    #features = StandardScaler().fit_transform(features);
    
            param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];

            param_grid = [
                {'svc__C' : param_range,
                'svc__kernel' : ['linear']},
                {
                    'svc__C': param_range,
                    'svc__gamma' : param_range,
                    'svc__kernel' : ['rbf']
                }
            ];

            parameter_space = {
            'mlp__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,),(50,),(25,),(10,)],
            'mlp__activation': ['tanh', 'relu'],
            'mlp__solver': ['sgd', 'adam'],
            'mlp__alpha': [0.0001,0.001,0.01,0.1],
            'mlp__learning_rate': ['constant','adaptive', 'invscaling'],
        }

            parameter_space_gbc = {
                'gbc__n_estimators': [10,50,100,200,500],
                'gbc__learning_rate': [0.001,0.01,0.1,1],
                'gbc__max_depth':[1,2,3,4,5]
            }

            pipe = Pipeline([('scalar',RobustScaler()), ('svc',SVC(class_weight='balanced'))]);

            # Best idx: 3347  best score: 0.8095606555922161  best comb: (0, 1, 2, 3, 6, 9, 20, 21)
            # best_param: {'mlp__activation': 'tanh', 'mlp__alpha': 0.1, 'mlp__hidden_layer_sizes': (50,), 'mlp__learning_rate': 'invscaling', 'mlp__solver': 'adam'}
            # iris = datasets.load_iris()
            # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            # svc = SVC()
            # clf = GridSearchCV(svc, parameters, scoring='f1', refit=True, n_jobs=-1)
            # clf.fit(iris.data, iris.target, )
            temp_cv = deepcopy(custom_cv);
            svm = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1', n_jobs=-1, cv = temp_cv);
            #mlp = GridSearchCV(estimator=MLPClassifier(), param_grid=parameter_space, scoring='f1', cv = 10, refit=True, n_jobs=-1);
            #ada = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=parameter_space_ada, scoring='f1', cv = 10, refit=True, n_jobs=-1);

            #lbl_list = preprocessing.label_binarize(lbl_list, classes=[0,1,2]);

            svm = svm.fit(features, total_y);
            #mlp = mlp.fit(total_data, lbl_list);

            # vc = VotingClassifier(estimators=[('svm', svm), ('mlp', mlp), ('ada', ada)], voting='hard');
            #scores = cross_val_score(svm, total_data, lbl_list, scoring='f1', cv=10)
            #print(scores.mean());

            if svm.best_score_ > best_score:
                best_idx = i;
                best_param = svm.best_params_;
                best_score = svm.best_score_;
                print('\n*********************\n');
                print(f"Best idx: {best_idx}\tbest score: {best_score}\tbest comb: {ps[best_idx]}");
                print('\n*********************\n');
            print('\n=============\n');
    print(f"Best idx: {best_idx}\tbest score: {best_score}\tbest comb: {ps[best_idx]}\nbest_param: {best_param}");

#Best idx: 2772  best score: 0.7813729298273577  best comb: (0, 2, 3, 4, 6, 20, 13)
#best_param: {'gbc__learning_rate': 0.1, 'gbc__max_depth': 3, 'gbc__n_estimators': 100}

def preload_dataset():
    gt_data_df = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    gt_img_list = list(gt_data_df['Image']);
    gt_img_list = list(map(str,gt_img_list));
    sym_list = list(gt_data_df['Symmetric Hemithoraces']);
    gt_exp_list = list(map(str, gt_data_df['Exposure']));

    image_list = [];
    mask_list = [];
    lbl_list = [];
    exp_list = [];

    for idx in (range(len(gt_img_list))):

        meta_data = pickle.load(open(f"C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{gt_img_list[idx]}.meta", 'rb'));
        exp_lbl = gt_exp_list[gt_img_list.index(gt_img_list[idx])];
        exp_list.append(exp_lbl);

        lbl = sym_list[idx];
        if lbl == 1:
            lbl = 0;
        elif lbl == 2:
            lbl = 1;
    
        lbl_list.append(lbl);
        spine_mask = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs', 'labels', meta_data['Spine'][2]), cv2.IMREAD_GRAYSCALE);
        spine_mask = np.where(spine_mask > 0, 1, 0).astype(np.bool8);
        ribs_mask = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs', 'labels', meta_data['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
        ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.bool8);
        mask = np.zeros(shape=(ribs_mask.shape[0], ribs_mask.shape[1], 1));
        mask[spine_mask] = 2;
        mask[ribs_mask] = 1;
        mask = np.int32(mask);

        pickle.dump(mask, open(f'cache\\{gt_img_list[idx]}.msk', 'wb'));
        mask_list.append(f'cache\\{gt_img_list[idx]}.msk');
        image_list.append(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final', f"{gt_img_list[idx]}.jpeg"));

        
    image_list = np.array(image_list);
    mask_list = np.array(mask_list);
    lbl_list = np.array(lbl_list);
    exp_list = np.array(exp_list);

    return store_folds(image_list, mask_list, lbl_list, exp_list);

def preload_dataset_test_1():

    gt_data_df = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    s_list, as_list = pickle.load(open('sampled_data.dt', 'rb'));
    gt_img_list = list(gt_data_df['Image']);
    gt_img_list = list(map(str,gt_img_list));

    image_list = [];
    mask_list = [];
    lbl_list = [];

    for idx, m in tqdm(enumerate(s_list)):
        meta_data = pickle.load(open(f"C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{m}.meta", 'rb'));
        lbl_list.append(0);
        spine_mask = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs', 'labels', meta_data['Spine'][2]), cv2.IMREAD_GRAYSCALE);
        spine_mask = np.where(spine_mask > 0, 1, 0).astype(np.bool8);
        ribs_mask = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs', 'labels', meta_data['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
        ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.bool8);
        mask = np.zeros(shape=(ribs_mask.shape[0], ribs_mask.shape[1], 1));
        mask[spine_mask] = 2;
        mask[ribs_mask] = 1;
        mask = np.int32(mask);

        pickle.dump(mask, open(f'cache\\{m}.msk', 'wb'));
        mask_list.append(f'cache\\{m}.msk');
        image_list.append(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final', f"{m}.jpeg"));
    
    for idx, m in tqdm(enumerate(as_list)):
        meta_data = pickle.load(open(f"C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{m}.meta", 'rb'));
        lbl_list.append(1);
        spine_mask = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs', 'labels', meta_data['Spine'][2]), cv2.IMREAD_GRAYSCALE);
        spine_mask = np.where(spine_mask > 0, 1, 0).astype(np.bool8);
        ribs_mask = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs', 'labels', meta_data['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
        ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.bool8);
        mask = np.zeros(shape=(ribs_mask.shape[0], ribs_mask.shape[1], 1));
        mask[spine_mask] = 2;
        mask[ribs_mask] = 1;
        mask = np.int32(mask);

        pickle.dump(mask, open(f'cache\\{m}.msk', 'wb'));
        mask_list.append(f'cache\\{m}.msk');
        image_list.append(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final', f"{m}.jpeg"));


        
    image_list = np.array(image_list);
    mask_list = np.array(mask_list);
    lbl_list = np.array(lbl_list);

    return store_folds(image_list, mask_list, lbl_list);

def store_folds(image_list, mask_list, lbl_list, exp_list):
    fold_cnt = 0;
    sfold = StratifiedKFold(5, shuffle=True, random_state=42);
    for train_id, test_id in sfold.split(image_list, lbl_list):
        train_x, train_mask, train_lbl, train_exp_lbl, test_x, test_mask, test_lbl, test_exp_lbl = image_list[train_id], mask_list[train_id], lbl_list[train_id], exp_list[train_id], image_list[test_id], mask_list[test_id], lbl_list[test_id], exp_list[test_id]
        pickle.dump([train_x, train_mask, train_lbl, train_exp_lbl, test_x, test_mask, test_lbl, test_exp_lbl], open(f'{fold_cnt}.dump','wb'))
        fold_cnt += 1;


def get_histogram(img, bins):
    temp_img = np.where(img == 255, 1, 0);
    h,w = img.shape;
    if h < bins:
        ph = bins;
        padded_img = np.zeros((ph,w));
        padded_img[:h,:] = img;
        img = padded_img;
        h = ph;

    rows_per_bin = int(h / bins);
    hist_horizontal = [];
    for i in range(0,h,rows_per_bin):
        s = temp_img[i:i+rows_per_bin,:];
        hist_horizontal.append(int(s.sum()));
    
    hist_horizontal = np.array(hist_horizontal, dtype=np.float32);
    hist_horizontal = np.expand_dims(hist_horizontal, axis=1);
    hist_horizontal = hist_horizontal / hist_horizontal.sum();

    hist_vertical = [];
    for i in range(0,w,rows_per_bin):
        s = temp_img[:,i:i+rows_per_bin];
        hist_vertical.append(int(s.sum()));
    
    hist_vertical = np.array(hist_vertical, dtype=np.float32);
    hist_vertical = np.expand_dims(hist_vertical, axis=1);
    hist_vertical = hist_vertical / hist_vertical.sum();
    
    return hist_horizontal, hist_vertical;

'''
    Sample images for Test #1
'''
def sample_images():
    df = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    img_list = list(map(str, df['Image']));
    exp_list = list(map(str, df['Exposure']));
    sym_list = list(df['Symmetric Hemithoraces']);
    sel = [];
    as_list = [];
    s_list = [];

    while (len(s_list) != 100 or len(as_list) != 100):
        r = np.random.randint(0, len(img_list));
        #select a unique random number
        while (r in sel):
            r = np.random.randint(0, len(img_list));
        e = exp_list[r];
        if e == '0.0':
            if (sym_list[r] == 0 or sym_list[r] == 1) and len(s_list) != 100:
                s_list.append(img_list[r]);
            elif (sym_list[r] == 2) and len(as_list) != 100:
                as_list.append(img_list[r]);
        sel.append(r);
    
    pickle.dump([s_list, as_list], open('sampled_data.dt', 'wb'));


if __name__ == "__main__":
    preload_dataset();

    build_experiment_dataset('obscured');
    build_thorax();
    compare_thorax_segmentation();

   # spine_and_ribs_segmentation_model = Unet(1).to(config.DEVICE);

    newtwork_trainer = NetworkTrainer();

    spine_and_ribs_segmentation_model = Unet(1).to(config.DEVICE);

    newtwork_trainer = NetworkTrainer();

    #optimize_symmetry_model();
    for curr_fold in range(5):
        print(f'===============Starting fold: {curr_fold}==================');
        fold_data = pickle.load(open(f'{curr_fold}.dump', 'rb'));
        train_x, train_mask, train_lbl, train_exp_lbl, test_x, test_mask, test_lbl, test_exp_lbl = fold_data[0], fold_data[1], fold_data[2], fold_data[3], fold_data[4], fold_data[5], fold_data[6], fold_data[7];
        spine_and_ribs_segmentation_model = newtwork_trainer.train('spine and ribs', 1, spine_and_ribs_segmentation_model, curr_fold, train_x, train_mask, train_exp_lbl, 
        test_x, test_mask, test_exp_lbl);

    optimize_symmetry_model();
    sample_count = 5;
    for count in range(sample_count):
        reg = 10**uniform(-6,-3);
        lr = 10**uniform(-3,-6);
        avg_f1 = [];
        for curr_fold in range(5):
            print(f'===============Starting fold: {curr_fold}==================');
            fold_data = pickle.load(open(f'{curr_fold}.dump', 'rb'));
            train_x, train_mask, train_lbl, test_x, test_mask, test_lbl = fold_data[0], fold_data[1], fold_data[2], fold_data[3], fold_data[4], fold_data[5];
            f1 = newtwork_trainer.train('spine and ribs', 3, spine_and_ribs_segmentation_model, curr_fold, train_x, train_mask, 
            test_x, test_mask,[reg,lr]);
            avg_f1.append(f1);
        print(f'Lr: {lr}\treg: {reg}\tf1: {np.mean(avg_f1)}');
        f = open(f'res{count}.txt', 'a');
        f.write(f'Lr: {lr}\treg: {reg}\tf1: {np.mean(avg_f1)}');
        f.close();


    spine_and_ribs_segmentation_model = UNET(1,config.IMAGE_SIZE,16,768,12,0.0,4,3).to(config.DEVICE);

    newtwork_trainer = NetworkTrainer();

    spine_and_ribs_segmentation_model = Unet(3).to(config.DEVICE);

    newtwork_trainer = NetworkTrainer();

    for curr_fold in range(5):
        print(f'===============Starting fold: {curr_fold}==================');
        fold_data = pickle.load(open(f'{curr_fold}.dump', 'rb'));
        train_x, train_mask, train_lbl, train_exp_lbl, test_x, test_mask, test_lbl, test_exp_lbl = fold_data[0], fold_data[1], fold_data[2], fold_data[3], fold_data[4], fold_data[5], fold_data[6], fold_data[7];
        spine_and_ribs_segmentation_model = newtwork_trainer.train('spine and ribs', 3, spine_and_ribs_segmentation_model, curr_fold, train_x, train_mask, train_exp_lbl, 
        test_x, test_mask, test_exp_lbl);

    for curr_fold in range(5):
        print(f'Starting fold: {curr_fold:#^10}');
        fold_data = pickle.load(open(f'{curr_fold}.dump', 'rb'));
        train_x, train_mask, train_lbl, train_exp_lbl, test_x, test_mask, test_lbl, test_exp_lbl = fold_data[0], fold_data[1], fold_data[2], fold_data[3], fold_data[4], fold_data[5], fold_data[6], fold_data[7];

        newtwork_trainer.train('spine and ribs', 3, spine_and_ribs_segmentation_model, curr_fold, train_x, train_mask, train_exp_lbl,
        test_x, test_mask,test_exp_lbl);
        avg_f1.append(f1);
    print(f'Lr: {lr}\treg: {reg}\tf1: {np.mean(avg_f1)}');
    f = open(f'res{count}.txt', 'a');
    f.write(f'Lr: {lr}\treg: {reg}\tf1: {np.mean(avg_f1)}');
    f.close();


