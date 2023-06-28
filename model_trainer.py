from copy import deepcopy
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils import shuffle
from numpy.core.fromnumeric import mean
from numpy.lib.npyio import load
from torch.nn.modules.loss import L1Loss
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
from utility import extract_sternum_features, postprocess_sternum, remove_outliers, scale_width
import config
from deep_learning.network_dataset import CanineDataset
from deep_learning.network import Unet
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
from PIL import Image
from glob import glob
from torchvision.utils import save_image
import albumentations as A
from sklearn.svm import SVC
import torchvision.transforms.functional as F
from ignite.contrib.handlers.tensorboard_logger import *
from torch.utils.data import DataLoader
from torchmetrics import *
#import ptvsd
from deep_learning.stopping_strategy import *
from deep_learning.loss import dice_loss, focal_loss, tversky_loss
from utility import divide_image_symmetry_line, get_symmetry_line, remove_blobs, remove_blobs_spine
from Symmetry.thorax import segment_thorax
from utils import create_folder, extract_caudal_features, extract_cranial_features, extract_symmetry_features, get_histogram, remove_outliers_hist_hor, remove_outliers_hist_ver
import matplotlib.pyplot as plt

def train_cranial_model(fold_cnt, train_features, train_lbl):
    params = {'svc__C': 0.001, 'svc__kernel': 'linear'}
    model = make_pipeline(RobustScaler(),
            SVC(C=params['svc__C'],  kernel = params['svc__kernel']));
    train_features = list(train_features);
    train_features = np.array(train_features);
    model.fit(train_features, np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\cranial_model.pt', 'wb'));
    return model;

def train_caudal_model(fold_cnt, train_features, train_lbl):
    params = {'svc__C': 1.0, 'svc__gamma': 1000.0, 'svc__kernel': 'rbf'}
    model = make_pipeline(RobustScaler(),
            SVC(C=params['svc__C'], kernel = params['svc__kernel'], gamma=params['svc__gamma'],));
    train_features = list(train_features);
    train_features = np.array(train_features);
    model.fit(train_features, np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\caudal_model.pt', 'wb'));
    return model;

def train_symmetry_model(fold_cnt, train_features, train_lbl):
    model = make_pipeline(RobustScaler(),
            SVC(C=1.0, kernel = 'rbf'));
    train_features = list(train_features);
    train_features = np.array(train_features);
    model.fit(train_features, np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\symmetry_model.pt', 'wb'));
    return model;

def train_sternum_model(fold_cnt, train_features, train_lbl):

    params = {'svc__C': 0.01, 'svc__gamma': 1.0, 'svc__kernel': 'rbf'}
    model = make_pipeline(RobustScaler(),
            SVC(class_weight='balanced', C=params['svc__C'], gamma=params['svc__gamma'], kernel = params['svc__kernel']));
    train_features = list(train_features);
    train_features = np.array(train_features);
    model.fit(np.expand_dims(train_features,axis = 1), np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\sternum_model.pt', 'wb'));
    return model;

def train_full_model(fold_cnt, train_features, train_lbl):
    params = {'gradientboostingclassifier__learning_rate': 0.001, 'gradientboostingclassifier__max_depth': 4, 'gradientboostingclassifier__n_estimators': 500}
    model = make_pipeline(RobustScaler(),
            GradientBoostingClassifier(learning_rate=params['gradientboostingclassifier__learning_rate'], max_depth=params['gradientboostingclassifier__max_depth'], 
            n_estimators = params['gradientboostingclassifier__n_estimators']));
    model.fit(train_features, np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\full_model.pt', 'wb'));
    return model;
    

def evaluate_test_data(fold_cnt, segmentation_models, classification_models, test_imgs, test_grain_lbl, test_lbl, transformer = None, use_saved_features = False):
    all_predictions = [];
    cnt = 0;
    #if use_saved_features is False:
    #    create_folder(f'results\\{fold_cnt}\\outputs', delete_if_exists=True);

    for idx,radiograph_image_path in (enumerate(test_imgs)):
        if use_saved_features is False:
            radiograph_image = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{radiograph_image_path}.jpeg'),cv2.IMREAD_GRAYSCALE);
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            radiograph_image = clahe.apply(radiograph_image);
            radiograph_image = np.expand_dims(radiograph_image, axis=2);
            radiograph_image = np.repeat(radiograph_image, 3,axis=2);


            transformed = config.valid_transforms(image = radiograph_image);
            radiograph_image = transformed["image"];
            radiograph_image = radiograph_image.to(config.DEVICE);
            
            #spine and ribs
            out = segmentation_models[0](radiograph_image.unsqueeze(dim=0));
            out = (torch.softmax(out, dim= 1)[0].permute(1,2,0)).detach().cpu().numpy();
            out = np.argmax(out,axis = 2);

            ribs = (out == 1).astype("uint8")*255;
            spine = (out == 2).astype("uint8")*255;

            #ribs = remove_blobs(ribs);

            # spine = remove_blobs_spine(spine).astype("uint8");
            # #----------------------------------------------------

            # #diaphragm
            # diaphragm = segmentation_models[1](radiograph_image.unsqueeze(dim=0));
            # diaphragm = torch.sigmoid(diaphragm)[0].permute(1,2,0).detach().cpu().numpy().squeeze();
            # diaphragm = diaphragm > 0.5;
            # diaphragm = np.uint8(diaphragm)*255;
            # #----------------------------------------------------

            # #sternum
            # sternum = segmentation_models[2](radiograph_image.unsqueeze(dim=0));
            # sternum = torch.sigmoid(sternum)[0].permute(1,2,0).detach().cpu().numpy().squeeze();
            # sternum = sternum > 0.7;
            # sternum = np.uint8(sternum)*255;
            #sternum = postprocess_sternum(sternum);
            # #----------------------------------------------------

            # #Symmetry
            
            # ribs = cv2.imread(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_ribs.png',cv2.IMREAD_GRAYSCALE);
            # hist_hor, hist_ver = get_histogram(ribs,1024);
            # ribs_new = remove_outliers_hist_ver(hist_ver, ribs);
            # ribs_new = remove_outliers_hist_hor(hist_hor, ribs_new);
            # koft = (ribs-ribs_new).sum();            
            # if koft > 150000:

            #     # fig, ax = plt.subplots(1,4);
            #     # ax[0].plot(hist_hor);
            #     # ax[1].plot(hist_ver);
            #     # ax[2].imshow(whole_thorax);
            #     # ax[3].imshow(whole_thorax_new);
            #     # #plt.savefig(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_clean.png')
            #     # plt.show();
            #     spine = cv2.imread(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_spine.png', cv2.IMREAD_GRAYSCALE);
            #     sym_line = get_symmetry_line(spine);
            #     ribs_left, ribs_right = divide_image_symmetry_line(ribs_new, sym_line);
            #     whole_thorax = segment_thorax(ribs_new);
            #     thorax_left = segment_thorax(ribs_left);
            #     thorax_right = segment_thorax(ribs_right);
            #     symmetry_features = extract_symmetry_features(thorax_left, thorax_right);
            #     symmetry_features = np.array(symmetry_features);
            
            # # #----------------------------------------------------

            # # #Cranial
            # cranial = spine - whole_thorax;
            # cranial_features = extract_cranial_features(cranial);
            # cranial_features = np.array(cranial_features);
            # # #-----------------------------------------------------

            # # #Caudal
            # caudal_features, diaphragm = extract_caudal_features(diaphragm, whole_thorax);
            # caudal_features = np.array(caudal_features)
            
            # # #-----------------------------------------------------

            # # #Sternum
            # spine_scaled = scale_width(spine,3).astype('uint8');
            # sternum = np.logical_and(sternum.squeeze(), np.where(whole_thorax>0, 1, 0)).astype(np.uint8);
            #sternum_features = extract_sternum_features(sternum, spine_scaled);

            # pickle.dump(cranial_features, open(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_cranial.feat','wb'));
            # pickle.dump(caudal_features, open(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_caudal.feat','wb'));
            #pickle.dump(symmetry_features, open(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_symmetry.feat','wb'));
        # pickle.dump(sternum_features, open(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_sternum.feat','wb'));

        # #store results
        # cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_spine.png', spine);
        # cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_spine_scaled.png', spine_scaled);
            cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_ribs_orig.png', ribs);
        # cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_diaph.png', diaphragm);
        # cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_sternum.png', sternum*255);
            # cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_thorax.png', whole_thorax);
            # cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_left.png', thorax_left);
            # cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_right.png', thorax_right);
            
        else:
            symmetry_features = pickle.load(open(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_symmetry.feat','rb'));
            cranial_features = pickle.load(open(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_cranial.feat','rb'));
            caudal_features = pickle.load(open(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_caudal.feat','rb'));
            sternum_features = pickle.load( open(f'results\\{fold_cnt}\\outputs\\{radiograph_image_path}_sternum.feat','rb'));
        
    #     cranial_lbl = classification_models[0].predict(cranial_features.reshape(1,-1));
    #     caudal_lbl = classification_models[1].predict(caudal_features.reshape(1,-1));
    #     symmetry_lbl = classification_models[2].predict(symmetry_features.reshape(1,-1));
    #     sternum_lbl = classification_models[3].predict(np.array(sternum_features[1]).reshape(1,-1));
            
        
    #     grain_lbls = [cranial_lbl[0], caudal_lbl[0],  sternum_lbl[0]];

    #     #grain_lbls = transformer.transform(np.array(grain_lbls).reshape(1,-1));
    #     #-----------------------------------------------------

    #     # cv2.imshow('spine', spine);
    #     # cv2.waitKey();

    #     quality_lbl = classification_models[4].predict(np.array(grain_lbls).reshape(1,-1)); 
    #     if quality_lbl[0] != test_lbl[idx]:
    #         print(f'grain: {grain_lbls} \ttrue grain: {test_grain_lbl[idx]}\tpred: {quality_lbl[0]}\ttrue: {test_lbl[idx]}');
    #         print(radiograph_image_path);

    #     all_predictions.append([cranial_lbl[0], caudal_lbl[0], sternum_lbl[0], quality_lbl[0]]);
    

    # #get performance metrics

    # all_predictions = np.array(all_predictions);
    # cranial_precision, cranial_recall, cranial_f1,_ = precision_recall_fscore_support(np.array(test_grain_lbl[:,0],np.int32), np.array(all_predictions[:,0],np.int32), average='binary');
    # cranial_accuracy = accuracy_score(np.array(test_grain_lbl[:,0],np.int32), np.array(all_predictions[:,0],np.int32));

    # caudal_precision, caudal_recall, caudal_f1,_ = precision_recall_fscore_support(np.array(test_grain_lbl[:,1],np.int32), np.array(all_predictions[:,1],np.int32), average = 'binary');
    # caudal_accuracy = accuracy_score(np.array(test_grain_lbl[:,1],np.int32), np.array(all_predictions[:,1],np.int32));

    # symmetry_precision, symmetry_recall, symmetry_f1,_ = precision_recall_fscore_support(np.array(test_grain_lbl[:,2],np.int32), np.array(all_predictions[:,2],np.int32), average = 'binary');
    # symmetry_accuracy = accuracy_score(np.array(test_grain_lbl[:,2],np.int32), np.array(all_predictions[:,2],np.int32));

    # sternum_precision, sternum_recall, sternum_f1,_ = precision_recall_fscore_support(np.array(test_grain_lbl[:,2],np.int32), np.array(all_predictions[:,2],np.int32), average='binary');
    # sternum_accuracy = accuracy_score(test_grain_lbl[:,2], all_predictions[:,2]);

    # quality_precision, quality_recall, quality_f1,_ = precision_recall_fscore_support(test_lbl, np.array(all_predictions[:,3],np.int32), average='binary');
    # quality_accuracy = accuracy_score(test_lbl, np.array(all_predictions[:,3],np.int32));
    # # #--------------------------------------------------

    # return [cranial_precision, cranial_recall, cranial_f1, cranial_accuracy],\
    #        [caudal_precision, caudal_recall, caudal_f1, caudal_accuracy],\
    #        [symmetry_precision, symmetry_recall, symmetry_f1, symmetry_accuracy],\
    #        [sternum_precision, sternum_recall, sternum_f1, sternum_accuracy],\
    #        [quality_precision, quality_recall, quality_f1, quality_accuracy];


class NetworkTrainer():

    def __init__(self):
        pass

    def __loss_func(self, output, gt):
        if self.num_classes > 1:
            f_loss = focal_loss(output, gt,  arange_logits=True, mutual_exclusion=True);
            t_loss = tversky_loss(output, gt, sigmoid=False, arange_logits=True, mutual_exclusion=True)
            return  t_loss + f_loss;
        else:
            f_loss = focal_loss(output, gt,  arange_logits=True, mutual_exclusion=False);
            t_loss = tversky_loss(output, gt, sigmoid=True, arange_logits=True, mutual_exclusion=False)
            return  t_loss + f_loss;
        
    def __train_one_epoch(self, epoch, loader, model, optimizer):
        epoch_loss = [];
        step = 0;
        update_step = 1;
        pbar = enumerate(loader);
        print(('\n' + '%10s'*2) %('Epoch', 'Loss'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (radiograph, mask) in pbar:
            radiograph, mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE)
            model.zero_grad(set_to_none = True);

            with torch.cuda.amp.autocast_mode.autocast():
                pred = model(radiograph);
                loss = self.__loss_func(pred, mask);

            self.scaler.scale(loss).backward();
            epoch_loss.append(loss.item());
            step += 1;

            if step % update_step == 0:
                self.scaler.step(optimizer);
                self.scaler.update();

            pbar.set_description(('%10s' + '%10.4g') %(epoch, np.mean(epoch_loss)));

    def __eval_one_epoch(self, epoch, loader, model):
        epoch_loss = [];
        total_prec = [];
        total_rec = [];
        total_f1 = [];
        total_acc = [];
        
        pbar = enumerate(loader);
        print(('\n' + '%10s'*6) %('Epoch', 'Loss', 'Prec', 'Rec', 'F1', 'Acc'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        with torch.no_grad():
            for i ,(radiograph, mask) in pbar:
                radiograph,mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE);

                pred = model(radiograph);
                loss = self.__loss_func(pred, mask);

                epoch_loss.append(loss.item());
                
                if self.num_classes > 1:
                    pred = (torch.softmax(pred, dim = 1)).permute(0,2,3,1);
                    pred = torch.argmax(pred, dim = 3);
                else:
                    pred = torch.sigmoid(pred) > 0.5;
                prec = self.precision_estimator(pred.flatten(), mask.flatten().long());
                rec = self.recall_estimator(pred.flatten(), mask.flatten().long());
                acc = self.accuracy_esimator(pred.flatten(), mask.flatten().long());
                f1 = self.f1_esimator(pred.flatten(), mask.flatten().long());
                
                
                total_prec.append(prec.item());
                total_rec.append(rec.item());
                total_f1.append(f1.item());
                total_acc.append(acc.item());

                pbar.set_description(('%10s' + '%10.4g'*5) % (epoch, np.mean(epoch_loss),
                np.mean(total_prec), np.mean(total_rec), np.mean(total_f1), np.mean(total_acc)))

        return np.mean(epoch_loss), np.mean(total_acc), np.mean(total_prec), np.mean(total_rec), np.mean(total_f1);


    def train(self, task_name, num_classes, model, fold_cnt, train_imgs, train_mask, test_imgs, test_mask, load_trained_model = False):

        if load_trained_model is True:
            model.load_state_dict(pickle.load( open(f'results\\{fold_cnt}\\{task_name}.pt', 'rb')));
            return model;

        self.num_classes = num_classes;
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler();
        self.precision_estimator = Precision(num_classes=num_classes, multiclass=False if num_classes ==1 else True).to(config.DEVICE);
        self.recall_estimator = Recall(num_classes=num_classes, multiclass=False if num_classes ==1 else True).to(config.DEVICE);
        self.accuracy_esimator = Accuracy(num_classes=num_classes, multiclass=False if num_classes ==1 else True).to(config.DEVICE);
        self.f1_esimator = F1Score(num_classes=num_classes, multiclass=False if num_classes ==1 else True).to(config.DEVICE);

        train_dataset = CanineDataset(train_imgs, train_mask, config.train_transforms);
        valid_dataset = CanineDataset(test_imgs, test_mask, config.valid_transforms);

        train_loader = DataLoader(train_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True);

        valid_loader = DataLoader(valid_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=False);

        model.reset_weights();
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5);

        stopping_strategy = CombinedTrainValid(0.7,2);

        best = 100;
        e = 1;
        best_model = None;
        best_prec = 0;
        best_recall = 0;
        best_f1 = 0;
        best_acc = 0;

        print(f'Started training task: {task_name}');

        while(True):
            model.train();
            self.__train_one_epoch(e, train_loader,model, optimizer);

            model.eval();
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.__eval_one_epoch(e, train_loader, model);

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.__eval_one_epoch(e, valid_loader, model);

            print(f"Epoch {e}\tLoss: {train_loss}\tPrecision: {train_precision}\tRecall: {train_recall}\tAccuracy: {train_acc}\tF1: {train_f1}");
            print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");


            if(valid_loss < best):
                print("New best model found!");
                best = valid_loss;
                best_model = deepcopy(model.state_dict());
                best_prec = valid_precision;
                best_recall = valid_recall;
                best_f1 = valid_f1;
                best_acc = valid_acc;

            if stopping_strategy(valid_loss, train_loss) is False:
                break;
            e += 1;
        f = open(f'results\\{fold_cnt}\\res_{task_name}.txt', 'w');
        f.write(f"Valid \tPrecision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}");
        f.close();
        pickle.dump(best_model, open(f'results\\{fold_cnt}\\{task_name}.pt', 'wb'));

        #load model with best weights to save outputs
        model.load_state_dict(best_model);
        return model;