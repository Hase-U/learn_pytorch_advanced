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

def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):

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

        #　optimizer初期化
        optimizer.zero_grad()

        # 順伝搬計算
        with torch.set_grad_enabled(phase=="train"):
          # 順伝播計算
          outputs = net(images)

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



