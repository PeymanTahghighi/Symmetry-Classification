from copy import deepcopy
import os
import numpy as np
import cv2
import config
from sklearn.preprocessing import StandardScaler

def get_symmetry_line(img):
    assert img.ndim == 2, "Image should be grayscale"
    
    w,h = img.shape;

    symmetry_line = np.zeros((w,2), dtype = np.int32);

    for i in range(w):
        first_cord = None;
        second_cord = None;
        for j in range(h):
            if img[i][j] != 0 and first_cord is None:
                first_cord = j;
            elif img[i][j] != 0:
                second_cord = j;

        if second_cord != None and first_cord != None:    
            symmetry_line[i] = ((i,(second_cord + first_cord) / 2));
    
    #check for missed points with zero values,
    #we use the average of ten points after and ten points before
    #as their value
    look_ahead = 10;
    for idx, s in enumerate(symmetry_line):
        if s[1] == 0:
            start_idx = idx;
            #attemp to estimte this value
            sum = 0;
            cnt_pos = 0;
            while(cnt_pos != look_ahead):
                if(symmetry_line[start_idx][1] != 0):
                    sum += symmetry_line[start_idx][1];
                    cnt_pos += 1;
                start_idx += 1;

                if start_idx >= w:
                    break;
            
            start_idx = idx;
            cnt_neg = 0;
            while(cnt_neg != look_ahead):
                if(symmetry_line[start_idx][1] != 0):
                    sum += symmetry_line[start_idx][1];
                    cnt_neg += 1;
                start_idx -= 1;

                if start_idx < 0:
                    break;
            sum /= cnt_neg + cnt_pos;

            symmetry_line[idx] = (idx, sum);
    
    return symmetry_line;

def divide_image_symmetry_line(img, sym_line):
    img_left = np.zeros(shape = (img.shape[0], img.shape[1]), dtype=np.uint8);
    img_right = np.zeros(shape = (img.shape[0], img.shape[1]), dtype=np.uint8);

    w,h = img.shape;

    for s in sym_line:
        for j in range(h):
            if j < s[1]:
                img_left[s[0], j] = img[s[0], j];
            else:
                img_right[s[0], j] = img[s[0], j];
    
    return img_left, img_right;

def remove_outliers(lst):
    ret_lst = [];
    q1 = np.quantile(lst, axis=0, q=0.25);
    q3 = np.quantile(lst, axis=0, q=0.75);
    iqr = q3-q1;
    dist_list = [];
    total_dist = 0;
    for idx, p in enumerate(lst):
        x_range_start = q1[0] - 1.5*iqr[0];
        x_range_end = q3[0] + 1.5*iqr[0];

        y_range_start = q1[1] - 1.5*iqr[1];
        y_range_end = q3[1] + 1.5*iqr[1];
        if p[0]<x_range_end and p[0]>x_range_start and p[1] < y_range_end and p[1] > y_range_start:
            ret_lst.append(idx);
    
    return ret_lst;

def remove_outliers_spine(lst):
    ret_lst = [];
    q1 = np.quantile(lst, axis=0, q=0.25);
    q3 = np.quantile(lst, axis=0, q=0.75);
    iqr = q3-q1;
    dist_list = [];
    total_dist = 0;
    for idx, p in enumerate(lst):
        x_range_start = q1[0] - 1.5*iqr[0];
        x_range_end = q3[0] + 1.5*iqr[0];

        y_range_start = q1[1] - 1.5*iqr[1];
        y_range_end = q3[1] + 1.5*iqr[1];
        if p[0]<x_range_end and p[0]>x_range_start and p[1] < y_range_end and p[1] > y_range_start:
            ret_lst.append(idx);

    return ret_lst;

def remove_blobs(ribs):
    kernel = np.ones((5,5), dtype=np.uint8);
    kernel_c = np.ones((35,35), dtype=np.uint8);
    opening = cv2.morphologyEx(ribs, cv2.MORPH_OPEN, kernel);
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_c);
    # cv2.imshow('open', opening);
    # cv2.imshow('close', closing);
    # cv2.waitKey();
    ret_img = np.zeros_like(closing);

    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    mean_area = 0;
    for c in contours:
        mean_area += cv2.contourArea(c);
    
    mean_area /= len(contours);

    position_list = [];
    all_position = [];
    all_area = [];
    for c in contours:
        area = cv2.contourArea(c);
        dia = cv2.arcLength(c, True);
        #list.append([area, dia]);
        x,y,w,h = cv2.boundingRect(c);
        center = [x+w/2,y+h/2];
        all_position.append(center);
        all_area.append([area, dia]);
    
    max_area = np.mean(all_area);
    positions = remove_outliers(all_position);
    
    q1 = np.quantile(all_area, 0.1, axis = 0);
    
    for idx, p in enumerate(contours):
        if all_area[idx][0] > max_area * 0.1:
            ret_img = cv2.fillPoly(ret_img, [contours[idx]], (255,255,255));
            
    return ret_img;

def remove_blobs_spine(ribs):
    kernel = np.ones((5,5), dtype=np.uint8);
    kernel_c = np.ones((41,41), dtype=np.uint8);
    opening = cv2.morphologyEx(ribs, cv2.MORPH_OPEN, kernel);
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_c);
    # cv2.imshow('open', opening);
    # cv2.imshow('close', closing);
    # cv2.waitKey();
    ret_img = np.zeros_like(closing);
    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    all_area = [];
    for c in contours:
        area = cv2.contourArea(c);
        all_area.append(area);

    biggest = np.max(all_area);
    for idx, a in enumerate(all_area):
        if a > 0.2*biggest:
            #simplify spine
            cvh = cv2.approxPolyDP(contours[idx], 10,True);
            ret_img = cv2.fillPoly(ret_img, [cvh], (255,255,255));

    return ret_img;

def find_data_mean(images):
    all_images = [];
    for i in images:
        file_name = os.path.basename(i);
        file_name = file_name[:file_name.rfind('.')];
        mask_full = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Full radiograph segmentation\\labels\\', f'{file_name}_0.png'), cv2.IMREAD_GRAYSCALE);
        mask_full = cv2.resize(mask_full,(config.IMAGE_SIZE, config.IMAGE_SIZE));
        mask_full = np.where(mask_full > 0, 1, 0);
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        radiograph_image = (cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE), (config.IMAGE_SIZE, config.IMAGE_SIZE)) * mask_full).astype("uint8");
        radiograph_image = clahe.apply(radiograph_image);
        all_images.append(radiograph_image);
    all_images = np.array(all_images);
    s = StandardScaler();
    s.fit(all_images.reshape(-1,config.IMAGE_SIZE**2));
    mean = s.mean_.reshape(config.IMAGE_SIZE, config.IMAGE_SIZE);
    
    var = s.var_.reshape(config.IMAGE_SIZE, config.IMAGE_SIZE);
    # for img in all_images:
    #     img = s.transform(img.reshape(-1,config.IMAGE_SIZE**2));
    #     print(np.max(img));
    #     print(np.min(img));

    # test = deepcopy(all_images[0]);
    # test = s.transform(test.reshape(-1,config.IMAGE_SIZE**2));

    # test2 = deepcopy(all_images[0]);
    # test2 = (test2 - mean)/(np.sqrt(var)+1e-6);

    # diff = np.sum(np.subtract(test,test2));
    # s = np.std(all_images,axis = 0) + 1e-6;
    return mean,np.sqrt(var)+1e-6;

def obscured_elipse(img, left, right):
    left = cv2.resize(left, (img.shape[1], img.shape[0]));
    contours = cv2.findContours(left, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    max_area = 0;
    max_cnt = None;
    for c in contours:
        area = cv2.contourArea(c);
        if area > max_area:
            max_area = area;
            max_cnt = c;
    
    elip = cv2.fitEllipse(max_cnt);
    
    tmp = np.zeros_like(left);
    tmp = cv2.ellipse(tmp,center= (int(elip[0][0]), int(elip[0][1])),axes=(int(elip[1][0]/3), int(elip[1][1]/4)), angle=int(elip[2]), startAngle=0, endAngle=360, color = (255,255,255), thickness=-1);

    tmp = np.where(tmp>0, 0, 1);
    img = tmp*img;

    right = cv2.resize(right, (img.shape[1], img.shape[0]));
    contours = cv2.findContours(right, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    max_area = 0;
    max_cnt = None;
    for c in contours:
        area = cv2.contourArea(c);
        if area > max_area:
            max_area = area;
            max_cnt = c;
    
    elip = cv2.fitEllipse(max_cnt);
    tmp = np.zeros_like(right);
    tmp = cv2.ellipse(tmp,center= (int(elip[0][0]), int(elip[0][1])),axes=(int(elip[1][0]/3), int(elip[1][1]/4)), angle=int(elip[2]), startAngle=0, endAngle=360, color = (255,255,255), thickness=-1);
    
    tmp = np.where(tmp>0, 0, 1);
    img = tmp*img;
    img = cv2.resize(img.astype("uint8"), (512,512));
    return img;

def get_max_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    
    if len(contours) == 0:
        return 0, 0;

    max_cnt = 0;
    max_area = 0;
    for c in contours:
        a = cv2.contourArea(c);
        if a > max_area:
            max_area = a;
            max_cnt = c;
    
    return max_cnt, max_area;