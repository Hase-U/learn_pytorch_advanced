import numpy as np
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
# import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET 
import cv2
from math import sqrt
from itertools import product
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import torch.nn.functional as F
import torch.nn.init as init

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

import sys, os
sys.path.append('./utils')
# import match これではモジュールを指してしまう
from match import match


def make_datapath_list(rootpath):

  # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
  imgpath_template = osp.join(rootpath, "JPEGImages", "%s.jpg")
  annopath_template = osp.join(rootpath, "Annotations", "%s.xml")

  #訓練と検証、それぞれのファイルのIDを取得する
  train_id_names = osp.join(rootpath+"ImageSets/Main/train.txt")
  val_id_names = osp.join(rootpath+"ImageSets/Main/val.txt")

  # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
  train_img_list = list()
  train_anno_list = list()

  for line in open(train_id_names):
    file_id = line.strip()
    img_path = (imgpath_template % file_id) #画像パス
    anno_path = (annopath_template % file_id) #アノテーションのパス
    train_img_list.append(img_path)
    train_anno_list.append(anno_path)

    #検証データの画像ファイルとアノテーションファイルへのパスリストの作成
    val_img_list = list()
    val_anno_list = list()

  for line in open(val_id_names):
    # print(file_id)
    file_id = line.strip()
    img_path = (imgpath_template % file_id) #画像パス
    anno_path = (annopath_template % file_id) #アノテーションのパス
    val_img_list.append(img_path)
    val_anno_list.append(anno_path)   
  
  """
  それぞれで何が起きてるか確認用
  """
  # print("line : ", line)
  # print("file_id : ",file_id)
  # print("annopath_template : ",annopath_template)
  # print("anno_path : ",anno_path)

  return train_img_list, train_anno_list, val_img_list, val_anno_list

# XML形式をリスト形式に変換する

class Anno_xml2list(object):
  #objectは今から入力する画像の中に存在し得る物体のリスト


  def __init__(self, classes):
    self.classes = classes

  def __call__(self, xml_path, width, height):

    # 画像内の全ての物体のアノテーションをこのリストに格納する
    res = []

    # xmlファイルを読み込む
    xml = ET.parse(xml_path).getroot()

    # 画像内にある物体(object)の数だけループする
    for obj in xml.iter("object"):
      
      #アノテーションで検知がdifficultに設定されているものは除外
      difficult = int(obj.find("difficult").text)
      if difficult == 1:
        continue
      
      #１つの物体に対するアノテーションを格納するリスト
      bndbox = []

      name = obj.find("name").text.lower().strip() #小文字揃えにして改行、空白をのぞいてる
      bbox = obj.find("bndbox") #バウンディボックスの情報

      # アノテーションのxmin, ymin, xmax, ymaxを取得して、[0,1]で正規化
      pts = ["xmin", "ymin", "xmax", "ymax"]

      for pt in pts:
        #VOCは原点が(1,1)なので(0,0)に
        cur_pixel = int(bbox.find(pt).text) - 1

        # 幅、高さで正規化
        if pt =="xmin" or pt == "xmax": 
          cur_pixel /= width   
        else:
          cur_pixel /= height

        bndbox.append(cur_pixel)
      
      # アノテーションのクラス名のindexを取得して追加
      label_idx = self.classes.index(name)
      bndbox.append(label_idx)

      #　resに["xmin", "ymin", "xmax", "ymax", "label_idx"]を加える
      res += [bndbox]

    return np.array(res)

from data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords,\
PhotometricDistort,Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DataTransform():


  def __init__(self, input_size, color_mean):
    self.data_transform = {
        "train":Compose([
                         ConvertFromInts(), # intをfloat32
                         ToAbsoluteCoords(), # アノテーションデータの規格化を戻す
                         PhotometricDistort(), # 画像の色調などをランダムに変更
                         Expand(color_mean), # 画像のキャンバスを広げる
                         RandomSampleCrop(), # 画像内の部分をランダムに抜き出す
                         RandomMirror(), #　画像を反転させる
                         ToPercentCoords(), # アノテーションデータを[0,1]に規格化
                         Resize(input_size), 
                         SubtractMeans(color_mean)
        ]),
      "val":Compose([
                     ConvertFromInts(), 
                     Resize(input_size),
                     SubtractMeans(color_mean)
      ])
    }
  
  def __call__ (self, img, phase, boxes, labels):

    return self.data_transform[phase](img, boxes, labels)

class VOCDataset(data.Dataset):

  def __init__(self, img_list, anno_list, phase, transform, transform_anno):
    self.img_list = img_list
    self.anno_list = anno_list
    self.phase = phase
    self.transform = transform # 画像の変形
    self.transform_anno = transform_anno #xml2lim



  def __len__(self):
    #画像の枚数を返す
    return len(self.img_list)

  def __getitem__(self, index):
    # 前処理した画像のテンソル形式のデータとアノテーションを取得

    im, gt, h, w = self.pull_item(index)
    return im, gt

  def pull_item(self, index):
    # 前処理した画像のテンソル形式のデータ、アノテーション、画像の高さ、幅を取得する

    #画像の読み込み
    image_file_path = self.img_list[index]
    img = cv2.imread(image_file_path)
    height, width, channels = img.shape

    # xml形式のアノテーション情報をリストに
    anno_file_path = self.anno_list[index]
    anno_list = self.transform_anno(anno_file_path, width, height)

    # 前処理を実施
    img, boxes, labels = self.transform(
        img, self.phase, anno_list[:,:4], anno_list[:,4])
    #色チャネルをBGRからRGBに変換
    img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

    #BBoxとラベルをセットしたnp.array
    gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

    """ 

    動作確認

    """

    # print("labels : ", labels)
    # print("np.expand_dims(labels, axis=1) : ", np.expand_dims(labels, axis=1))
    # print(gt)

    return img, gt, height, width

def od_collate_fn(batch):

  # batchはDatasetクラスの__getitem__のreturn

  # 本来batchはbatch_sizeの数だけimgがリストで並んでるが、今回は[img, gt]となってる

  targets = []
  imgs = []

  for  sample in batch:
    imgs.append(sample[0])
    targets.append(torch.FloatTensor(sample[1]))

  imgs = torch.stack(imgs, dim = 0)

  return imgs, targets

def make_vgg():
  layers = []
  in_channels = 3

  # 前半の５つのまとまりの部分の実装
  cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "MC",
         512, 512, 512, "M",  # ここまでの出力がsource1となる.
         512, 512, 512]

  for v in cfg:
    if v == "M":
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    elif v == "MC":
      layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode = True)]
      #ceil_mode：出力サイズの計算で少数部分を切り上げるか切り下げるか。デフォ(False)は切り下げる
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      layers += [conv2d, nn.ReLU(inplace = True)]
      in_channels = v # 次の層で用いる

  
  pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
  conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
  conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
  layers += [
             pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

  return nn.ModuleList(layers)

def make_extras():
  layers = []
  in_channels = 1024

  # extraモジュールの畳み込み層のチャネル数
  cfg = [256, 512, 128, 256, 128, 256, 128, 256]

  layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
  layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
  layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
  layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
  layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
  layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
  layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
  layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]

  return nn.ModuleList(layers)

def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):

  loc_layers = []
  conf_layers = []

  # source1
  loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]*4, # DBoxの情報は４つ
                           kernel_size=3, padding=1)]
  conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]*num_classes,
                           kernel_size=3, padding=1)]

  # source2
  loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]*4, # DBoxの情報は４つ
                           kernel_size=3, padding=1)]
  conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]*num_classes,
                           kernel_size=3, padding=1)]

  # source3
  loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]*4, # DBoxの情報は４つ
                           kernel_size=3, padding=1)]
  conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]*num_classes,
                           kernel_size=3, padding=1)]

  # source4
  loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]*4, # DBoxの情報は４つ
                           kernel_size=3, padding=1)]
  conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]*num_classes,
                           kernel_size=3, padding=1)]  

  # source5
  loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]*4, # DBoxの情報は４つ
                           kernel_size=3, padding=1)]
  conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]*num_classes,
                           kernel_size=3, padding=1)] 

  # source6
  loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]*4, # DBoxの情報は４つ
                           kernel_size=3, padding=1)]
  conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]*num_classes,
                           kernel_size=3, padding=1)]                                       
    
  return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

# scale=20のL2Normで正則化する
class L2Norm(nn.Module):
  def __init__(self, input_channels = 512, scale = 20):
    super(L2Norm, self).__init__()
    self.weight = nn.Parameter(torch.Tensor(input_channels))
    self.scale = scale
    self.reset_parameters()
    self.eps = 1e-10

  def reset_parameters(self):
    #結合パラメータを大きさscaleの値にする初期化
    nn.init.constant_(self.weight, self.scale) # self.weightの値がscale(=20)に変換される

  def forward(self, x):
    norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
    x = torch.div(x, norm) # xの各要素をnormで割る。0だと困るのでself.epsを足しとく

    weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
    #配列の次元をunsqueezeで合わせて実際の要素数はxと同じにする 
    out = weights*x

    
    return out

# デフォルトボックスを出力するクラス
class DBox(object):
  def __init__(self, cfg):
    super(DBox, self).__init__()

    # 初期設定
    self.image_size = cfg["input_size"] # 画像サイズの300
    # 各sourceの特徴量マップサイズ
    self.feature_maps = cfg["feature_maps"]
    self.num_priors = len(cfg["feature_maps"]) # sourceの個数=6


    """
    ここらへんの値は絶妙な値を与えられてる？
    """
    self.steps = cfg["steps"] # DBoxのピクセルサイズ
    self.min_sizes = cfg["min_sizes"] # 小さい正方形のDBoxのピクセルサイズ
    self.max_sizes = cfg["max_sizes"]


    self.aspect_ratios = cfg["aspect_ratios"] # 長方形のアスペクト比

  def make_dbox_list(self):
    # DBoxを作成する
    mean = []

    for k, f in enumerate(self.feature_maps):
      for i, j in product(range(f), repeat=2):

        #　特徴量の画像サイズ
        f_k = self.image_size / self.steps[k]

        # DBoxの中心。　ただし[0,1]で規格化
        cx = (j + 0.5)/ f_k
        cy = (i + 0.5)/ f_k

        #　アスペクト比1の小さいDBox [cx, cy, width, heighy]
        s_k = self.min_sizes[k]/self.image_size
        mean += [cx, cy, s_k, s_k]

        # アスペクト比１の大きなDBox 
        s_k_prime = sqrt(s_k*(self.max_sizes[k]/self.image_size))
        mean += [cx, cy, s_k_prime, s_k_prime]

        # その他のアスペクト比のdefBox
        for ar in self.aspect_ratios[k]:
          mean += [cx, cy, s_k*sqrt(ar), s_k*sqrt(ar)]
          mean += [cx, cy, s_k/sqrt(ar), s_k/sqrt(ar)]
        
    # DBoxをテンソルに変換 
    output = torch.Tensor(mean).view(-1,4)

    # 大きさを[0,1]におさめる
    output.clamp_(max=1, min = 0)

    return output

def decode(loc, dbox_list):

  """
  loc: [8732,4]
    SSDモデルで推論するオフセット情報

  dbox_list: [8732,4]
    DBoxの情報

  # Return
  boxes : [xmin,ymin,xmax,ymax]
  """

  # オフセット情報からBBoxを求める
  boxes = torch.cat((
      dbox_list[:, :2] + loc[:, :2]*0.1 + dbox_list[:,:2],
      dbox_list[:, 2:] + torch.exp(loc[:, 2:]*0.2)), dim = 1)
  
  # (cx,cy,w,h)から(xmin,ymin,xmax,ymax)に変える
  boxes[:, :2] -= boxes[:, 2:]/2 
  boxes[:, 2:] += boxes[:, :2]

  return boxes

# Non-Maximum Suppresionを行う関数
def nm_suppression(boxes, scores, overlap=0.45, top_k=200):

  """
  Para
  ------
  boxes : [確信度の閾値を超えたBBoxの数, 4] (xmin, ymin, xmax, ymax)

  scores : [確信度の閾値を超えたBBoxの数] (確信度)
  -----

  Return
  --------
  keep : list,  nmsを通過したindexが降順格納

  count : int,  nmsを通過したBBoxの数
  --------
  """

  # returnのひな形を形成
  count = 0
  keep = scores.new(scores.size(0)).zero_().long() # scoresと同じ大きさで要素が０のtorch


  # 各BBoxの面積
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]
  area = torch.mul(x2 - x1, y2 - y1)

  # boxesをコピーする。あとでBBoxの被り度合いIOUの計算に利用する
  tmp_x1 = boxes.new() # .new()したら全部一緒では
  tmp_y1 = boxes.new() # .new()の中にtorch.size()を入れると同じ形の要素が0に十分近いものになる
  tmp_x2 = boxes.new()
  tmp_y2 = boxes.new()
  tmp_h = boxes.new()
  tmp_w = boxes.new()

  # scoreを昇順に並び替える
  v, idx = sores.sort(0)

  # 上位top_k個(=200)のBBoxのindexを取り出す
  idx = idx[-top_k:]

  # idxの要素数が0になるまでwhile
  while idx.numel() > 0:
    i = idx[-1] # max

    # keepにiを格納し、indexからこれとBBoxを指すidを抜いていく
    keep[count] = i
    count += 1

    # 最後BBoxだったら以下を飛ばす
    if idx.size(0) == 1:
      break
    
    # 現在のiの分をidxから削除
    idx = idx[: -1]

    # ここから今格納したBBoxと被りが大きいBBoxを削除する
    torch.index_select(x1, 0, idx, out=tmp_x1) #x1のtensorを、0dimの情報が保たれるように、idxで指定された場所だけtmp_x1に格納して
    torch.index_select(y1, 0, idx, out=tmp_y1)
    torch.index_select(x2, 0, idx, out=tmp_x2)
    torch.index_select(y2, 0, idx, out=tmp_y2)

    # 全てのBBoxに対して、現在のBBox=indexがiと被っている値までに設定(clamp:最大最小の範囲を限定する)
    tmp_x1 = torch.clamp(tmp_x1, min = x1[i])
    tmp_y1 = torch.clamp(tmp_y1, min = y1[i])
    tmp_x2 = torch.clamp(tmp_x2, max = x2[i])
    tmp_y2 = torch.clamp(tmp_y2, max = y2[i])

    # wとhをiを除いたテンソルと同じ形にする
    tmp_w.resize_as_(tmp_x2)
    tmp_h.resize_as_(tmp_y2)

    # clampで最大最小を制限した状態でBBoxの幅と高さを求める
    tmp_w = tmp_x2 - tmp_x1
    tmp_h = tmp_y2 - tmp_y1

    # 幅や高さが負だったら0
    tmp_w = torch_clamp(tmp_w, min = 0.0)
    tmp_h = torch_clamp(tmp_h, min = 0.0)

    # clampされた状態で面積を求める
    inter = tmp_w*tmp_h  # ANDの部分の面積を求めた

    # IoU = ANDの部分/ORの部分
    rem_areas = torch.index_select(area, 0, idx)
    union = (rem_areas - inter) + area[i]
    IoU = inter/union

    # IoUがoverlapよりidxを除く
    idx = idx[IoU.le(overlap)] # le : Less than or Equal to
    # idx = idx[idx >= overlap]



  return keep, count


# import torch.autograd.Function as Function

class Detect(torch.autograd.Function):

  def __init__(self, conf_thresh=0.01, top_k = 200, nms_thresh=0.45):
    self.softmax = nn.Softmax(dim=-1) # confをソフトマックスで正規化
    self.conf_thresh = conf_thresh
    self.top_k = top_k
    self.nms_thresh = nms_thresh

  def forward(self, loc_data, conf_data, dbox_list):
    """
    Return
    torch.Size([batch_nu,, 21, 200, 5])
    (batch_num, クラス、　confのtop200, BBox情報)
    """

    # 各サイズを取得
    num_batch = loc_data.size(0) #ミニバッチサイズ
    num_dbox = loc_data.size(1)
    num_classes = loc_data.size(2)

    # ソフトマックスによる正規化
    conf_data = self.softmax(conf_data)

    # 出力の箱を用意
    output = torch.zeros(num_batch, num_classes, self.top_k, 5)

    # cof_dataを[batch_size, 8732, num_classes] から [batch_size,  num_classes, 8732]
    conf_preds = conf_data.transpose(2,1)

    # ミニバッチごとにループ
    for i in range(num_batch):

      # locとDBoxからBBox
      decoded_boxes = decode(loc_data[i], dbox_list)

      # confのコピーを作成
      conf_scores = conf_preds[i].clone()

      # 画像クラスごとのループ(index=0は背景なので飛ばす)
      for cl in range(1, num_classes):
        
        # maskの作成
        c_mask = conf_scores[cl].gt(self.conf_thresh) #gt :Greater Than

        # scores : torch.Size([閾値を超えたBBoxの数])
        scores = conf_scores[cl][c_mask]

        # 閾値を超えたconfがない場合
        if scores.nelement() == 0:
          continue
        
        # decoded_boxesも閾値を超えたものだけにしたい
        l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
        boxes = decoded_boxes[l_mask].view(-1,4) #そのままだと一次元になってしまうので.viewを使って変形

        # Non-Maximumを適用
        ids, count = nm_suppression(
            boxes, scores, self.nms_thresh, self.top_k)
        
        # outputの対応する箇所に結果を格納する
        output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                           boxes[ids[:count]]),1)

    return output

# SSDクラスを作成する
class SSD(nn.Module):

  def __init__(self, phase, cfg):
    super(SSD, self).__init__()
  
    self.phase = phase
    self.num_classes = cfg["num_classes"]

    # SSDネットワークを作成
    self.vgg = make_vgg()
    self.extras = make_extras()
    self.L2Norm = L2Norm()
    self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])


    #DBoxの作成
    dbox = DBox(cfg)
    self.dbox_list = dbox.make_dbox_list()

    # 推論時はクラス[Detect]を用意
    if phase == "inference":
      self.detect = Detect()


  """
  ------------------------

  ここからが新しい実装

  ------------------------
  """

  def forward(self, x):
    sources = list()
    loc = list()
    conf = list()

    # source1まで計算
    for k in range(23):
      x = self.vgg[k](x)
    
    # source1の出力をL2正則化してsource1を完成させて格納
    source1 = self.L2Norm(x)
    sources.append(source1)

    # source2作成
    for k in range(23, len(self.vgg)):
      x = self.vgg[k](x)

    sources.append(x)

    # extrasのconvとReLUを計算
    # source3~6
    for k, v in enumerate(self.extras):
      x = F.relu(v(x), inplace=True)
      if k % 2 == 1:
        sources.append(x)


    # source1~6について畳み込みを一回実施
    for (x, l , c) in zip(sources, self.loc, self.conf):
      #.permuteは要素入れ替え
      #sourceによってアスペクト比の数が異なるのでアスペクト比の種類の項を一番後ろに持っていく
      #viewをあとで使いたいから.contiguous()でメモリ上で連続配置させる
      loc.append(l(x).permute(0,2,3,1).contiguous())
      conf.append(c(x).permute(0,2,3,1).contiguous())

    # loc, conf自体はリスト型であることに注意
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    # print("conf size: ", conf.size())

    # さらに整形
    loc = loc.view(loc.size(0), -1, 4)
    conf = conf.view(conf.size(0), -1, self.num_classes)

    # 最後に出力する
    output = (loc, conf, self.dbox_list)

    if self.phase =="inference":
      return self.detect(output[0], output[1], output[2])
    else:
      return output
    
class MultiBoxLoss(nn.Module):
  # SSDの損失関数のためのクラス

  def __init__(self, jaccard_thresh=0.5, neg_pos=3, device="cpu"):
    super(MultiBoxLoss, self).__init__()
    self.jaccard_thresh = jaccard_thresh
    self.negpos_ratio = neg_pos
    self.device = device

  def forward(self, predictions, targets):

    """
    損失関数の計算を行う

    Parameters
    ----------------
    predictions : tuple, netの訓練時の出力 (loc,conf,dbox_list)

    targets : [num_batch, num_objs, 5]
      5は[xmin,,,,+正解ラベル]
    
    Returns
    -----------
    loss_l : Tensor, locの損失関数
    loss_c : Tensor, confの損失関数

    """

    # SSDnetの出力がタプルなので分解
    loc_data, conf_data, dbox_list = predictions
    # print(conf_data.size())

    # 要素数を把握
    num_batch = loc_data.size(0)
    num_dbox = loc_data.size(1)
    num_classes = conf_data.size(2)

    # 損失の計算に用いる
    # それぞれDBoxから一番近いBBoxの情報を格納している。
    conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
    loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

    # matchの出力結果をconf_tとloc_tの対応する場所に格納していく
    for idx in range(num_batch):

      # 現在のidxにおける正解を取り出す
      truths = targets[idx][:, :-1].to(self.device) # [xmin, ymin, xmax, ymax]
      labels = targets[idx][:, -1].to(self.device)  # 正解ラベル

      # デフォルトボックスを新しく用意
      dbox = dbox_list.to(self.device)

      # DBoxからBBoxにする時に使用
      variance = [0.1, 0.2]

      match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)
      
    
    """
    ここから　loss_l を計算する。
    ただし背景クラスを割り当てられたものは計算対象から除く
    """ 
    pos_mask = conf_t_label > 0
    
    #  pos_maskをloc_dataのmaskとして使えるように変形
    pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

    # loc_pとloc_tを得る
    loc_p = loc_data[pos_idx].view(-1, 4) 
    loc_t = loc_t[pos_idx].view(-1,4) # 教師

    loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')


    """ 
    ここから loss_cを計算する
    ただしHard　Negative Miningを実施する
    背景クラスDBoxと予測したものので損失が大きいものを学習に用いる
    """
    batch_conf = conf_data.view(-1, num_classes)
    # print(batch_conf.size())
    # print(conf_t_label.size())
    # print(conf_t_label.view(-1)[2000:3000])

    # クラス予測の損失を計算。reduction="none"にして和を取らない
    loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction="none")

    # Negative DBoxに対してHard Negative Miningを施す
    num_pos = pos_mask.long().sum(1, keepdim=True) # ミニバッチごとのオブジェクトの数

    loss_c = loss_c.view(num_batch, -1) # torch.Size([num_batch, 8732])
    loss_c[pos_mask] = 0 # obujectのあるDBoxは損失を0にする

    # Hard Negative Mining を実施する
    # 各DBox の損失の大きさloss_cの順位であるidx_rankを求める
    _, loss_idx = loss_c.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)

    # 学習に用いる背景DBoxの数を決める
    # 全体のDBox(8732)を超える場合にはそれを上限にする
    num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)

    # 背景DBoxの数 num_negより損失の順位が高い(idのidx_rankの数字的には小さい) DBoxをとるmask
    neg_mask = idx_rank < (num_neg).expand_as(idx_rank)
    """
    num_neg = 100だったら[100, 100, ...]となってるので
    idx_rank = [3, 102, 34, ...]となっていたら neg_mask = [True, False, True, ...]となる
    """

    # それぞれのマスクを整形する
    pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
    neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

    # conf_dataに含まれる背景DBoxの中で使わないものを除く
    conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes)
    #gt(0)はいずれかのmaskによって要素が1となっているところを取り出すため

    # labelも同様に取り出す
    conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]
    # print(conf_hnm.size())
    # print(conf_t_label_hnm.size())

    # sumを求める
    loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction="sum")

    # 物体を発見したBBoxの数でNで割る
    N = num_pos.sum()
    loss_l /= N
    loss_c /= N

    return loss_l, loss_c

  


"""
DataLoaderの作成
"""

# 2-2 ファイルパスのリストを取得
rootpath = "data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

# 2-2 Datasetを作成
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

color_mean = (104, 117, 123)
input_size = 300

train_dataset = VOCDataset(train_img_list, train_anno_list, phase = "train",
                           transform=DataTransform(input_size, color_mean),
                           transform_anno = Anno_xml2list(voc_classes))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase = "val",
                           transform=DataTransform(input_size, color_mean),
                           transform_anno = Anno_xml2list(voc_classes))

# 2-3 DataLoaderを作成する
batch_size = 32

train_dataloader = data.DataLoader(
    train_dataset, batch_size = batch_size, shuffle = True, collate_fn = od_collate_fn)

val_dataloader = data.DataLoader(
    val_dataset, batch_size = batch_size, shuffle = False, collate_fn = od_collate_fn)

#辞書型にまとめる
dataloaders_dict = {
    "train":train_dataloader, "val": val_dataloader}



"""
ネットワークモデルの作成
"""

# 2-4 SSD300の設定
ssd_cfg = {
    'num_classes': 21,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類  #これ使ってなくない？
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める 
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

# SSDネットワークモデル
net = SSD(phase="train", cfg=ssd_cfg)

# SSDの初期の重みを設定
vgg_weights = torch.load('weights/vgg16_reducedfc.pth')
net.vgg.load_state_dict(vgg_weights)

# その他のネットをHeの初期値で初期化
def weights_init(m):
  if isinstance(m, nn.Conv2d):
    init.kaiming_normal_(m.weight.data)
    if m.bias is not None: # バイアス項がある場合
      nn.init.constant_(m.bias, 0.0)

net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス : ", device)

print("ネットワーク設定完了")


"""
損失関数と最適化手法の設定
"""

# 損失関数の設定
criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

# 最適化手法の設定
optimizer = optim.SGD(net.parameters(), lr=1e-3,
                      momentum=0.9, weight_decay=5e-4)




"""
学習と検証
"""

# モデルを学習させる関数の作成

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

  # GPUが使えるかの確認
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("使用デバイス : ", device)

  net.to(device)
  torch.backends.cudnn.benchmark = True # 高速化

  # イテレーションカウンタをセット
  iteration = 1
  epoch_train_loss = 0.0
  epoch_val_loss = 0.0
  logs = []

  # epochのループ
  for epoch in range(num_epochs+1):

    # 開始時間
    t_epoch_start = time.time()
    t_iter_start = time.time()

    print("-------------------------")
    print("Epoch {}/{} ".format(epoch+1, num_epochs))
    print("-------------------------")

    # epochごとの訓練と検証のループ
    for phase in ["train", "val"]:
      if phase == "train":
        net.train()
        print(" (train) ")
      else:
        if((epoch+1)%10==0):
          net.eval()
          print("--------------------")
          print(" (val) ")
        else:
          continue
      

      # データローダーからミニバッチを取り出すループ
      for images, targets in dataloaders_dict[phase]:

        # GPUが使えるならGPUに送信
        images = images.to(device)
        targets = [ann.to(device) for  ann in targets]
        # print(images.size())
        # print(len(targets))

        #　optimizer初期化
        optimizer.zero_grad()

        # 順伝搬計算
        with torch.set_grad_enabled(phase=="train"):
          # 順伝播計算
          outputs = net(images)
        #   print(len(outputs))


          # 損失の計算
          loss_l, loss_c = criterion(outputs, targets)
          loss = loss_l + loss_c

          # 訓練時はバックプロパゲーション
          if phase == "train":
            loss.backward()

            # 勾配の大きさを制約
            nn.utils.clip_grad_value_(
                net.parameters(), clip_value=2.0)
            
            optimizer.step()

            if(iteration % 10 == 0):
              t_iter_finish = time.time()
              duration = t_iter_finish - t_iter_start
              print("イテレーション {} || Loss: {: .4} || 10iter: {:.4} sec. ".format(iteration, loss.item(), duration))
              t_iter_start = time.time()

            
            epoch_train_loss += loss.item()
            iteration += 1

          else:
            epoch_val_loss += loss.item()

    # epochのphaseごとのlossと正解率
    t_epoch_finish = time.time()
    print("-----------------")
    print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss, epoch_val_loss))
    print("timer: {: .4f} sec".format(t_epoch_finish - t_epoch_start))
    t_epoch_start = time.time()

    # ログを保存
    log_epoch = {
        "epoch": epoch+1,
        "train_loss": epoch_train_loss,
        "val_loss": epoch_val_loss}

    logs.append(log_epoch)
    df = pd.DataFrame(logs)
    df.to_csv("log_out.csv")

    epoch_train_loss = 0.0
    epoch_val_loss = 0.0

    # ネットワークを保存する
    if ((epoch+1)%10==0):
      torch.save(net.state_dict(), "weights/ssd300_"+str(epoch+1) + ".pth")



# 実行
# 訓練データ5717枚, 検証データ5823枚
num_epochs = 10
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)



