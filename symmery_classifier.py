#===========================================================
#===========================================================
from copy import deepcopy
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
import albumentations as A
import pickle
from pystackreg import StackReg
from tqdm import tqdm
from thorax import segment_thorax
import pandas as pd
from utility import divide_image_symmetry_line, get_symmetry_line
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from itertools import chain, combinations

#===========================================================
#===========================================================
EPSILON = 1e-5

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

ROOT = 'C:\\PhD\\Miscellaneous\\Spine and Ribs\\labels'
names = ['602 (1)', '620', '606', '636', '659 (1)', '669', '660', '744', 'DV12', '317']
def preload_data():
    all_masks = glob(f'{ROOT}\\*.meta');
    hemithoraces_dict = dict();
    for t in tqdm(all_masks):
        file_name = os.path.basename(t);
        file_name = file_name[:file_name.rfind('.')];
        meta_data = pickle.load(open(os.path.join(ROOT, f'{t}'), 'rb'));

        if 'Spine' in meta_data.keys() and 'Ribs' in meta_data.keys():
            spine_mask = cv2.imread(os.path.join(ROOT,  meta_data['Spine'][2]), cv2.IMREAD_GRAYSCALE);
            spine_mask = np.where(spine_mask > 0, 255, 0).astype(np.uint8);
            ribs_mask = cv2.imread(os.path.join(ROOT, meta_data['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
            ribs_mask = np.where(ribs_mask > 0, 255, 0).astype(np.uint8);

            sym_line = get_symmetry_line(spine_mask);
            ribs_left, ribs_right = divide_image_symmetry_line(ribs_mask, sym_line);
            thorax_left = segment_thorax(ribs_left);
            thorax_right = segment_thorax(ribs_right);

            hemithoraces_dict[file_name] = [thorax_left, thorax_right];

    return hemithoraces_dict;


def preprocess_train_dataset(hemithoraces_dict):
    #process folds data
    image_list, mask_list, lbl_list = pickle.load(open('all_data.dmp', 'rb'));
    for i in range(5):
        train_idxs = pickle.load(open(f'{i}.dmp', 'rb'))[0];
        test_idxs = pickle.load(open(f'{i}.dmp', 'rb'))[1];

        train_imgs = image_list[train_idxs];
        for t in train_imgs:
            file_name = os.path.basename(t);
            file_name = file_name[:file_name.rfind('.')];
            meta_data = pickle.load(open(os.path.join(ROOT, f'{file_name}.meta'), 'rb'));

            if 'Spine' in meta_data.keys() and 'Ribs' in meta_data.keys():

                if os.path.exists(f'{i}\\train') is False:
                    os.mkdir(f'{i}\\train');

                cv2.imwrite(f'{i}\\train\\{file_name}_left.png', hemithoraces_dict[file_name][0]);
                cv2.imwrite(f'{i}\\train\\{file_name}_right.png', hemithoraces_dict[file_name][1]);



def crop_top(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    #assert len(contours) == 1, "Number of contours detected should be exactly one";
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
    
    # for idx in range(len(hull_list)):
    #     img = cv2.drawContours(img, hull_list, idx, (255,255,255), 2);

    

    # cv2.imshow('ch', img);
    # cv2.imshow('chcrop', img_crop);
    # cv2.waitKey();
    
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
    #out = util.to_uint16(out);
    #out = np.where(out !=0, 1, 0);
    # cv2.imshow('o', out);
    # cv2.waitKey();
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
    p = p + EPSILON;
    q = q + EPSILON;
    avg = (p+q)/2;
    jsd = (cross_entropy(p,avg) - cross_entropy(p,p))/2 + (cross_entropy(q,avg) - cross_entropy(q,q))/2;
    #clamp
    if jsd > 1.0:
        jsd = 1.0;
    elif jsd < 0.0:
        jsd = 0.0;
    
    return jsd;

def extract_features(img_left, img_right):
    img_crop_left = crop_top(img_left);
    img_crop_right = crop_top(img_right);
    w,h = img_left.shape;
    area_left = (img_left == 255).sum() / (w*h);
    area_left_crop = (img_crop_left == 255).sum() / (img_crop_left.shape[0]*img_crop_left.shape[1]);
    peri_left, peri_left_crop = get_perimeter(img_left);

    area_right = (img_right == 255).sum() / (w*h);
    area_right_crop = (img_crop_right == 255).sum() / (img_crop_right.shape[0]*img_crop_right.shape[1]);
    peri_right, peri_right_crop = get_perimeter(img_right);

    f = area_left / area_right;
    f_crop = area_left_crop / area_right_crop;
    # else:
    #     f = area_right / area_left;
    
    s = 0;
    # if peri_left > peri_right:
    s = peri_left / peri_right;
    s_crop = peri_left_crop / peri_right_crop;
    #else:
    #s = peri_right / peri_left;

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
    #print(f"{img_list[i]}: {jsd}")
    #total_data.append(np.concatenate([jsd1, jsd2, diff1, diff2, diff3, f, s], axis=0));

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
    feat.append(hist_left_hor);
    feat.append(hist_right_hor);
    feat.append(hist_right_crop_hor);
    feat.append(hist_left_crop_hor);

    return feat;


if __name__ == "__main__":
    #hemithoraces = preload_data();
    #preprocess_train_dataset(hemithoraces);   
    
    # gt_file = pd.read_excel('G:\\My Drive\\dvvd_list_final.xlsx');
    # img_list = list(gt_file['Image']);
    # img_list = list(map(str, img_list));

    # gt_lbl = list(gt_file['Symmetric Hemithoraces']);
    # total_X = [];
    # total_Y = [];
    # train_fold_indices = [];
    # test_fold_indices = [];
    # for i in tqdm(range(5)):
        
    #     test_list = glob(f'{i}\\test\\*');
    #     train_list = glob(f'{i}\\train\\*');
    #     train_fold_indices.append(np.arange(len(total_X), len(total_X)+int(len(train_list)/2)));
    #     test_fold_indices.append(np.arange(len(total_X)+int(len(train_list)/2), len(total_X)+int(len(train_list)/2)+int(len(test_list)/2)));
        
    #     for idx in range(0,len(train_list)-1,2):
    #         file_name = os.path.basename(train_list[idx]);
    #         file_name = file_name[:file_name.rfind('_')];
    #         if file_name in img_list:
    #             img_indx = img_list.index(file_name);
    #             lbl = gt_lbl[img_indx];
    #             if lbl == 2:
    #                 lbl = 1;
    #             else:
    #                 lbl = 0;
    #             total_Y.append(lbl);
    #         else:
    #             print(file_name);
    #         img_left = cv2.imread(train_list[idx], cv2.IMREAD_GRAYSCALE);
    #         img_right = cv2.imread(train_list[idx+1], cv2.IMREAD_GRAYSCALE);

    #         feat = extract_features(img_left, img_right);

            
    #         total_X.append(feat);
        
    #     for idx in range(0,len(test_list)-1,2):
    #         file_name = os.path.basename(test_list[idx]);
    #         file_name = file_name[:file_name.rfind('_')];
    #         if file_name in img_list:
    #             img_indx = img_list.index(file_name);
    #             lbl = gt_lbl[img_indx];
    #             if lbl == 2:
    #                 lbl = 1;
    #             else:
    #                 lbl = 0;
    #             total_Y.append(lbl);
    #         else:
    #             print(file_name);
    #         img_left = cv2.imread(test_list[idx], cv2.IMREAD_GRAYSCALE);
    #         img_right = cv2.imread(test_list[idx+1], cv2.IMREAD_GRAYSCALE);

    #         feat = extract_features(img_left, img_right);

            
    #         total_X.append(feat);
    
    # custom_cv = zip(train_fold_indices, test_fold_indices);
    # pickle.dump([total_X,total_Y, custom_cv], open('data.dmp', 'wb'));

    data = pickle.load(open('data.dmp', 'rb'));
    total_x, total_y, custom_cv = data[0], data[1], data[2];


    total_x = np.array(total_x);
    total_y = np.array(total_y);

    ps = list(powerset(np.arange(0,19)));
    #lbl_list = np.expand_dims(lbl_list, axis = 1);
    best_idx = -1;
    best_score = 0;
    for i in tqdm(range(1, len(ps))):
        if len(ps[i]) > 1:
            features = np.zeros((total_x.shape[0], len(ps[i])));
            for j in range(len(ps[i])):
                features[:,j] = total_x[:,ps[i][j]];


            features = StandardScaler().fit_transform(features);
            
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
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }

            parameter_space_ada = {
                'n_estimators': [10,20,30,40,50,60,70,80,90,100]
            }

            pipe = Pipeline([('scalar',StandardScaler()), ('svc',SVC(class_weight='balanced'))]);

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
                best_score = svm.best_score_;
                print('\n*********************\n');
                print(f"Best idx: {best_idx}\tbest score: {best_score}\tbest comb: {ps[best_idx]}");
                print('\n*********************\n');
    print('\n=============\n');
    print(f"Best idx: {best_idx}\tbest score: {best_score}\tbest comb: {ps[best_idx]}");
    
