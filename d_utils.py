import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain

from skimage import measure
from sklearn import metrics

import torch
import torch.optim
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image

MVTEC_ITEMTYPES = ['carpet',
             'grid',
             'leather',
             'tile',
             'wood',
             'bottle',
             'cable',
             'capsule',
             'hazelnut',
             'metal_nut',
             'pill',
             'screw',
             'toothbrush',
             'transistor',
             'zipper',
             ]

MVTEC_PATH = "mvtec_data"

TRAIN_BATCH_SIZE = 32

IMAGE_TRANSFORM = transforms.Compose([
      transforms.Resize([256, 256]),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

class MVTecDataset(object):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return [Image.open(p).convert('RGB') for p in self.image_list]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx])
        return self.image_list[idx], image

class ResNet18_MS3(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet18_MS3, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res

def load_gt(root, cls):
    gt = []
    gt_dir = os.path.join(root, cls, 'ground_truth')
    sub_dirs = sorted(os.listdir(gt_dir))
    for sb in sub_dirs:
        for fname in sorted(os.listdir(os.path.join(gt_dir, sb))):
            temp = cv2.imread(os.path.join(gt_dir, sb, fname), cv2.IMREAD_GRAYSCALE)
            temp = cv2.resize(temp, (256, 256)).astype(bool)[None, ...]
            gt.append(temp)
    gt = np.concatenate(gt, 0)
    return gt
 
def test(teacher, student, loader):
    teacher.eval()
    student.eval()
    loss_map = np.zeros((len(loader.dataset), 64, 64))
    i = 0
    for batch_data in loader:
        _, batch_img = batch_data
        batch_img = batch_img.cuda()
        with torch.no_grad():
            t_feat = teacher(batch_img)
            s_feat = student(batch_img)
        score_map = 1.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            # aggregate score map by element-wise product
            score_map = score_map * sm
        loss_map[i: i + batch_img.size(0)] = score_map.squeeze().cpu().data.numpy()
        i += batch_img.size(0)
    return loss_map

def train_val(teacher, student, train_loader, val_loader, args, model_name='best'):
    min_err = 10000
    teacher.eval()
    student.train()

    optimizer = torch.optim.SGD(student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)
    for epoch in range(args.epochs):
        student.train()
        for batch_data in train_loader:
            _, batch_img = batch_data
            batch_img = batch_img.cuda()

            with torch.no_grad():
                t_feat = teacher(batch_img)
            s_feat = student(batch_img)

            loss =  0
            for i in range(len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], dim=1)
                s_feat[i] = F.normalize(s_feat[i], dim=1)
                loss += torch.sum((t_feat[i] - s_feat[i]) ** 2, 1).mean()

            print('[%d/%d] loss: %f' % (epoch, args.epochs, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        err = test(teacher, student, val_loader).mean()
        print('Validation Loss: {:.7f}'.format(err.item()))
        if err < min_err:
            min_err = err
            # save_name = os.path.join(args.model_save_path, args.category, 'best.pth.tar')
            save_name = os.path.join(args.model_save_path, args.category, f"{model_name}.pth.tar")
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            state_dict = {
                'category': args.category,
                'state_dict': student.state_dict()
            }
            torch.save(state_dict, save_name)

class DummyArgs():
    pass

def evaluate(labels, scores, metric='roc', integration_limit=None):
    if integration_limit is None:
        integration_limit = 0.3
    if metric == 'pro':
        return get_pro(labels, scores, integration_limit)
    if metric == 'roc':
        return get_roc(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")

def get_roc(labels, scores):
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def get_pro(masks, scores, integration_limit):
    '''
        https://github.com/YoungGod/DFR/blob/a942f344570db91bc7feefc6da31825cf15ba3f9/DFR-source/anoseg_dfr.py#L447
    '''
    # per region overlap
    max_step = 4000
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):
            label_map = measure.label(masks[i], connectivity=2)
            props = measure.regionprops(label_map, binary_score_maps[i])
            for prop in props:
                pro.append(prop.intensity_image.sum() / prop.area)
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr
        masks_neg = ~masks
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    ## default 30% fpr vs pro, pro_auc
    # expect_fpr = 0.3
    
    # Edited to make the integration limit a parameter
    expect_fpr = integration_limit
    idx = fprs <= expect_fpr    # # rescale fpr [0, 0.3] -> [0, 1]
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = rescale(pros_mean[idx])    # need scale
    pro_auc_score = metrics.auc(fprs_selected, pros_mean_selected)
    return pro_auc_score

class MNet(nn.Module):

  def __init__(self, pretrained=False):
    super(MNet, self).__init__()
    mnet = timm.create_model('mobilenetv2_100', pretrained=pretrained)
    self.return_nodes = [f'blocks.{i}' for i in range(1,6)]
    self.model = create_feature_extractor(mnet,
                 return_nodes=self.return_nodes) 

  def forward(self, x):
    out_dict = self.model(x)
    out_list = [out_dict[name] for name in self.return_nodes]
    return out_list

class Convnext(nn.Module):

  def __init__(self, pretrained=False):
    super(Convnext, self).__init__()
    cnet = timm.create_model('convnext_base_in22k', pretrained=pretrained)
    self.return_nodes = [f'stages.{i}' for i in range(4)]
    self.model = create_feature_extractor(cnet,
                                          return_nodes=self.return_nodes) 

  def forward(self, x):
    outs_dict = self.model(x)
    outs_list = [outs_dict[name] for name in self.return_nodes]
    return outs_list
  
def get_discrepancy_means_stds(teacher, student, train_loader, val_loader):

  teacher.eval()
  student.eval()

  with torch.no_grad():
    discs = []
    for _,x in chain(train_loader, val_loader):
      x = x.to('cuda')

      yhat_teacher = teacher(x)
      yhat_student = student(x)

      disc = [F.normalize(t, dim=1) - F.normalize(s, dim=1)
              for (t,s) in zip(yhat_teacher,yhat_student)]
      discs += [disc]

    layers = len(discs[0]) # Number of designated layers, typically 3

    concd = [torch.cat([d[i] for d in discs]) for i in range(layers)]
    means = [d.mean(dim=0) for d in concd]
    stds = [d.std(dim=0) for d in concd]
    return means, stds

def get_train_val_loaders(category,
                          mvtec_path=MVTEC_PATH,
                          batch_size=TRAIN_BATCH_SIZE):
  transform = IMAGE_TRANSFORM
  image_list = sorted(glob(os.path.join(mvtec_path, category, 'train', 'good', '*.png')))
  train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
  train_dataset = MVTecDataset(train_image_list, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
  val_dataset = MVTecDataset(val_image_list, transform=transform)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
  return train_loader, val_loader

def test_with_args(args):
  category = args.category
  teacher = ResNet18_MS3(pretrained=True)
  student = ResNet18_MS3(pretrained=False)
  teacher.cuda()
  student.cuda()

  checkpoint = f"snapshots/{category}/best.pth.tar"
  saved_dict = torch.load(checkpoint)
  gt = load_gt(args.data_dir, category)

  print('load ' + checkpoint)
  student.load_state_dict(saved_dict['state_dict'])

  np.random.seed(0)
  torch.manual_seed(0)
      
  transform = transforms.Compose([
      transforms.Resize([256, 256]),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  test_neg_image_list = sorted(glob(os.path.join(args.data_dir, category, 'test', 'good', '*.png')))
  test_pos_image_list = set(glob(os.path.join(args.data_dir, category, 'test', '*', '*.png'))) - set(test_neg_image_list)
  test_pos_image_list = sorted(list(test_pos_image_list))
  test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
  test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
  test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
  test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)

  pos = test(teacher, student, test_pos_loader)
  neg = test(teacher, student, test_neg_loader)
  
  scores = []
  for i in range(len(pos)):
      temp = cv2.resize(pos[i], (256, 256))
      scores.append(temp)
  for i in range(len(neg)):
      temp = cv2.resize(neg[i], (256, 256))
      scores.append(temp)

  scores = np.stack(scores)
  neg_gt = np.zeros((len(neg), 256, 256), dtype=bool)
  gt_pixel = np.concatenate((gt, neg_gt), 0)
  gt_image = np.concatenate((np.ones(pos.shape[0], dtype=bool), np.zeros(neg.shape[0], dtype=bool)), 0)        

  # Integration limit for AUROC not implemented yet
  # but included here for future use
  auc_pixel = evaluate(gt_pixel.flatten(),
                       scores.flatten(),
                       metric='roc',
                       integration_limit=args.integration_limit)
  auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')

  if args.scaling:
    print(f"These results are calculated WITH discrepancy scaling, eps={args.eps}.")
  else:
    print(f"These results are calculated WITHOUT discrepancy scaling.")

  if args.pro:
    print("Evaluating the PRO metric, this will take some time.")
    pro = evaluate(gt_pixel,
                   scores,
                   metric='pro',
                   integration_limit=args.integration_limit)
    print('Category: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(category, auc_pixel, auc_image_max, pro))
  else:
    print('Category: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}'.format(category, auc_pixel, auc_image_max))

  return (auc_pixel, auc_image_max, pro if args.pro else None)

def test_model(model, loader):
  model.eval()
  loss_map = np.zeros((len(loader.dataset), 64, 64))
  i = 0
  for batch_data in loader:
    _, batch_img = batch_data
    batch_img = batch_img.cuda()
    model_out = model(batch_img)
    loss_map[i: i + batch_img.size(0)] = model_out.squeeze().cpu().data.numpy()
    i += batch_img.size(0)
  return loss_map

def test_model2(model, loader):
  # Same as test_model, but does not preallocate the loss_map
  model.eval()
  loss_map = []
  for batch_data in loader:
    _, batch_img = batch_data
    batch_img = batch_img.cuda()
    model_out = model(batch_img)
    loss_map.append(model_out.cpu().data.numpy())
  return np.concatenate(loss_map, 0).squeeze()

def get_neg_pos_test_loaders(args, transform=IMAGE_TRANSFORM):
  category = args.category
  test_neg_image_list = sorted(glob(os.path.join(args.data_dir, category, 'test', 'good', '*.png')))
  test_pos_image_list = set(glob(os.path.join(args.data_dir, category, 'test', '*', '*.png'))) - set(test_neg_image_list)
  test_pos_image_list = sorted(list(test_pos_image_list))
  test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
  test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
  test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
  test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)
  return test_neg_loader, test_pos_loader

def test_with_model_and_args(model, args):
  model.eval()
  category = args.category
  gt = load_gt(args.data_dir, category)

  np.random.seed(0)
  torch.manual_seed(0)
      
  test_neg_loader, test_pos_loader = get_neg_pos_test_loaders(args, IMAGE_TRANSFORM)

  pos = test_model(model, test_pos_loader)
  neg = test_model(model, test_neg_loader)

  scores = []
  for i in range(len(pos)):
      temp = cv2.resize(pos[i], (256, 256))
      scores.append(temp)
  for i in range(len(neg)):
      temp = cv2.resize(neg[i], (256, 256))
      scores.append(temp)

  scores = np.stack(scores)
  neg_gt = np.zeros((len(neg), 256, 256), dtype=bool)
  gt_pixel = np.concatenate((gt, neg_gt), 0)
  gt_image = np.concatenate((np.ones(pos.shape[0], dtype=bool), np.zeros(neg.shape[0], dtype=bool)), 0)        

  # Integration limit for AUROC not implemented yet
  # but included here for future use
  auc_pixel = evaluate(gt_pixel.flatten(),
                       scores.flatten(),
                       metric='roc',
                       integration_limit=args.integration_limit)
  auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')

  if args.pro:
    print("Evaluating the PRO metric, this will take some time.")
    pro = evaluate(gt_pixel,
                   scores,
                   metric='pro',
                   integration_limit=args.integration_limit)
    print('Category: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(category, auc_pixel, auc_image_max, pro))
  else:
    print('Category: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}'.format(category, auc_pixel, auc_image_max))

  return (auc_pixel, auc_image_max, pro if args.pro else None)

class STM(nn.Module):

    def __init__(self,
                 teacher,
                 student,
                 means=None,
                 stds=None,
                 eps=0.05,
                 aggregator="prod"):
        super(STM, self).__init__()     
        self.teacher = teacher
        self.student = student
        self.means = means
        self.stds = stds
        self.eps = eps
        if aggregator not in ["prod", "sum"]:
            raise ValueError(f"Invalid aggregator {aggregator}.")
        self.aggregator = aggregator

    def forward(self, x):
        t_feat = self.teacher(x)
        s_feat = self.student(x)
        score_map = 0.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            diff = t_feat[j] - s_feat[j]
            if self.means is not None:
              diff -= self.means[j]
            
            if self.stds is None:
              pass
            elif isinstance(self.eps, float): 
              diff /= self.stds[j] + self.eps
            else:
              diff_tmp = 0.
              for eps in self.eps:
                diff_tmp += (diff / (self.stds[j] + eps)) / len(self.eps)
              diff = diff_tmp
            
            sm = torch.sum(diff ** 2, 1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            if self.aggregator == "sum":
                score_map = score_map + sm
            else:
                score_map += torch.log(sm)
        return score_map

class NewEnsemble(nn.Module):
  # Ensemble of models
  # Initialize with a list of models and an optional list of weights
  def __init__(self, models, weights=None, val_loader=None):
    super(NewEnsemble, self).__init__()
    for model in models:
        model.eval()
    self.models = models
    self.weights = weights
    if weights is None:
      self.weights = torch.tensor([1/len(models)] * len(models), dtype=torch.float32)
    assert len(self.weights) == len(self.models)
    # If validation loader is provided, compute mean and std of the outputs
    # of the models on the validation set
    with torch.no_grad():
      if val_loader is not None:
        self.means = []
        self.stds = []
        for model in self.models:
          outputs = []
          for _, x in val_loader:
            outputs.append(model(x.to('cuda')))
          outputs = torch.cat(outputs)
          self.means.append(outputs.mean())
          self.stds.append(outputs.std())
      else:
        self.means = [0.] * len(self.models)
        self.stds = [1.] * len(self.models)

  def forward(self, x):
    # Compute the weighted average of the normalized outputs of the models
    y = 0
    for w, model, mean, std in zip(self.weights, self.models, self.means, self.stds):
      normalized_output = (model(x) - mean) / (std + 1e-5)
      y += w * normalized_output
    return y

def load_STM(checkpoint_path,
             model_type='resnet18',
             category=None,
             aggregator="prod",
             force_recalc=False):
  '''Loads an STM from saved weights. To use discrepancy scaling,
     set the product category.'''
  if model_type in ['mobilenet', 'mnet']:
    teacher = MNet(pretrained=True)
    student = MNet(pretrained=False)
  elif model_type in ['convnext', 'conv']:
    teacher = Convnext(pretrained=True)
    student = Convnext(pretrained=False)
  elif model_type in ['resnet18', 'resnet_18', 'resnet']:
    teacher = ResNet18_MS3(pretrained=True)
    student = ResNet18_MS3(pretrained=False)
  else:
    raise ValueError(f"Unknown model type {model_type}.")
  teacher.eval()
  student.eval()
  teacher.cuda()
  student.cuda()

  # checkpoint = f"snapshots/{category}/best.pth.tar"
  saved_dict = torch.load(checkpoint_path)
  student.load_state_dict(saved_dict['state_dict'], strict=False)

  if category is not None:
    # If saved_dict contains discrepancy means and stds, use those
    if ('means' in saved_dict and 'stds' in saved_dict) and not force_recalc:
      means = saved_dict['means']
      stds = saved_dict['stds']
    else:
      train_loader, val_loader = get_train_val_loaders(category)
      means, stds = get_discrepancy_means_stds(teacher,
                                             student,
                                             train_loader,
                                             val_loader)
      # Save means and stds to checkpoint
      saved_dict['means'] = means
      saved_dict['stds'] = stds
      torch.save(saved_dict, checkpoint_path)
    stm = STM(teacher, student, means, stds, aggregator=aggregator)
  else:
    stm = STM(teacher, student, aggregator=aggregator)
  return stm
