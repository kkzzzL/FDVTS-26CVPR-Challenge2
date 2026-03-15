import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import sys
from utils import *
from tqdm import tqdm
from dataset1 import Lung3D_eccv_patient_supcon
from torch.utils.data import DataLoader
import torch.nn.functional as F
from visualize import Visualizer
from torchnet import meter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from models.ResNet import SupConResNet
#import segmentation_models_pytorch as smp
#from efficientnet_pytorch_3d import EfficientNet3D
from torch.utils.data import WeightedRandomSampler
#from timm.scheduler.cosine_lr import CosineLRScheduler

import torch.backends.cudnn as cudnn
import random
import math

from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

print("torch = {}".format(torch.__version__))

IMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--visname', '-vis', default='try_lr4', help='visname')
parser.add_argument('--batch_size', '-bs', default=8, type=int, help='batch_size')
parser.add_argument('--lr', '-lr', default=1e-4, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n_cls', default=4, type=int, help='n_classes')
parser.add_argument('--pretrain', '-pre', default=False, type=bool, help='use pretrained')
parser.add_argument('--supcon', '-con', default=False, type=bool, help='use supcon')
parser.add_argument('--mixup', '-mix', default=False, type=bool, help='use mix')
parser.add_argument('--box_lung', '-box_lung', default=False, type=bool, help='data box lung')
parser.add_argument('--seg_sth', '-seg_something', default=None, type=str, help='lung or lesion, cat to input')
parser.add_argument('--over_sampling', '-over_sampling', default=False, type=bool, help='over sampling')

parser.add_argument('--iccv_test', '-iccv_test', default=False, type=bool, help='use iccv test as train')
parser.add_argument('--weighted_loss', '-wl', default=False, type=bool, help='weighted ce loss')
parser.add_argument('--mosmed', '-mm', default=False, type=bool, help='use mosmed in challenge 2')
parser.add_argument('--model', '-model', default='resnest50_3D', type=str, help='use mosmed in challenge 2')
# 'c2dresnet50', 'medicalnet_resnet50', 'medicalnet_resnet34', 'resnest50_3D', 'P3DCResNet50'
parser.add_argument('--val_certain_epoch', '-val_certain_epoch', default=False, type=str, help='use mosmed in challenge 2')
parser.add_argument('--optimizer', '-optim', default='adam', type=str, help='use mosmed in challenge 2')


best_f1 = 0
val_epoch = 1
save_epoch = 10

my_whole_seed = 0
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False 

def parse_args():
    global args
    args = parser.parse_args()

def get_lr(cur, epochs):
    if cur < int(epochs * 0.3):
        lr = args.lr
    elif cur < int(epochs * 0.8):
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    return lr

def get_dynamic_lr(cur, epochs):
    power = 0.9
    lr = args.lr * (1 - cur / epochs) ** power
    return lr


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    device = x.get_device()
    if use_cuda:
        index = torch.randperm(batch_size).to(device).long()
    else:
        index = torch.randperm(batch_size).long()

    mixed_x = (lam * x + (1 - lam) * x[index,:]).clone()
    y_a, y_b = y, y[index]
    return mixed_x, y_a.long(), y_b.long(), lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1048576

def warmup_lr(optimizer, current_epoch, warmup_epochs, base_lr):
    lr = base_lr * float(current_epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    print(torch.cuda.device_count())
    global best_f1
    global save_dir

    parse_args()
    log_dir = os.path.join("runs", args.visname + "_" + IMESTAMP)
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard log dir:", log_dir)
    
    if args.seg_sth:
        ipt_dim=2
    else:
        ipt_dim=1
    # prepare the model
    
    target_model = SupConResNet(name=args.model, ipt_dim=ipt_dim, head='mlp', feat_dim=128, n_classes=2, supcon=args.supcon)
    if args.supcon:
        s1 = target_model.sigma1
        s2 = target_model.sigma2



    if args.n_classes == 4:
        if args.model == 'P3DCResNet50' or args.model == 'medicalnet_resnet50':
            target_model.encoder.classifier = nn.Linear(2048,4)
        elif args.model == 'medicalnet_resnet34':
            target_model.encoder.classifier = nn.Linear(512,4)
        else:
            target_model.encoder.fc = nn.Linear(2048,4)

    if args.pretrain:
        ckpt = torch.load('/remote-home/kejinlu/CVPR2026/challenge2/code/run/bak/eccv/28.pkl', map_location="cpu", weights_only=False)
        state_dict = ckpt['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        unParalled_state_dict.pop('encoder.fc.bias')
        unParalled_state_dict.pop('encoder.fc.weight')
        target_model.load_state_dict(unParalled_state_dict, strict=False)
    # for param in target_model.parameters():
    #     param.requires_grad = False
    
    # layers_to_train = [
    # 'head.0.weight',
    # 'head.0.bias',
    # 'head.2.weight',
    # 'head.2.bias'
    # ]
    
    # for name, param in target_model.named_parameters():
    #     if name in layers_to_train:
    #         param.requires_grad = True

    print('Params: ', count_parameters(target_model))

    target_model = nn.DataParallel(target_model)
    target_model = target_model.cuda()
    
    # prepare data
    train_data = Lung3D_eccv_patient_supcon(train=True,val=False,n_classes=args.n_classes, supcon=args.supcon, box_lung=args.box_lung, seg_sth=args.seg_sth, iccv_test=args.iccv_test, add_mosmed=args.mosmed)
    if args.over_sampling:
        group_list = []

        for item in train_data.datalist:

            img_path = item[0]

            # path example: train/G/female/xxx.npy
            parts = img_path.split('/')

            cls = parts[-3]
            gender = parts[-2]

            group = cls + "_" + gender
            group_list.append(group)

        # 统计每个 group 数量
        from collections import Counter
        group_count = Counter(group_list)

        print("group_count:", group_count)

        # 权重 = 1 / group_size
        weights = []
        for g in group_list:
            weights.append(1.0 / group_count[g])

        weights = torch.DoubleTensor(weights)

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True
        )
    
    val_data = Lung3D_eccv_patient_supcon(train=False,val=True,n_classes=args.n_classes, supcon=args.supcon, box_lung=args.box_lung, seg_sth=args.seg_sth, iccv_test=args.iccv_test, add_mosmed=args.mosmed)

    if args.over_sampling:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=8,pin_memory=True,drop_last=True)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8,pin_memory=True)

    criterion = SupConLoss(temperature=0.1)
    criterion = criterion.cuda()
    if args.n_classes==4:
        if args.weighted_loss:
            if args.mosmed:
                # weight = torch.tensor([0.0931, 0.0985, 0.1433, 0.6651]).cuda() #add mosmed
                weight = torch.tensor([0.618, 1.839, 0.772, 0.772]).cuda() #add mosmed
                # weight = torch.tensor([1., 1., 1., 2.]).cuda() #add mosmed
            else:
                # weight = torch.tensor([0.1506, 0.2065, 0.1506, 0.4923]).cuda()
                weight = torch.tensor([0.618, 1.839, 0.772, 0.772]).cuda()
                # weight = torch.tensor([1., 1., 1., 2.]).cuda()
        else:
            weight=None
    else:
        weight = None

    criterion_clf = nn.CrossEntropyLoss(weight=weight)
    criterion_clf = criterion_clf.cuda()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(target_model.parameters(), args.lr, weight_decay=1e-5)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, target_model.parameters()), args.lr, weight_decay=1e-5)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(target_model.parameters(), args.lr, momentum=0.9, weight_decay=1e-5)

    total_iters = args.epochs * len(train_loader)

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_iters=5 * len(train_loader),
        max_iters=total_iters,
        base_lr=args.lr
    )

    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/con/2026/'+ str(args.visname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  
    test_log=open('./logs/'+args.visname+'.txt','w')   

    if args.val_certain_epoch:
        weight_dir = 'checkpoints/con/grade_resnest50_con64_mix_catlung_1/58.pkl'
        epoch = int(weight_dir.split('/')[-1].split('.')[0])
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['net']
        target_model.load_state_dict(state_dict, strict=True) 

        val_log=open('./logs/val.txt','w')   
        
        val1(target_model,val_loader,epoch,val_log,args.optimizer, writer)
        exit()

    # train the model
    initial_epoch = 0
    for epoch in range(initial_epoch, args.epochs):
        target_model.train()
        con_matx.reset()
        total_loss1 = .0
        total_loss2 = .0
        total_loss3 = .0
        
        total = .0
        correct = .0
        count = .0
        total_num = .0

        # lr = args.lr
        #lr = get_lr(epoch, args.epochs)
        #for param_group in optimizer.param_groups:
            #param_group['lr'] = lr
        pred_list = []
        label_list = []
        
        pbar = tqdm(train_loader, ascii=True)
        for i, (imgs, masks, labels, ID) in enumerate(pbar):
            
            if args.supcon:
                imgs = torch.cat([imgs[0],imgs[1]],dim=0) #2*bsz,256,256
            imgs = imgs.unsqueeze(1).float().cuda() #2*bsz,1,256,256   
            # imgs = imgs.repeat(1,3,1,1,1)   

            if args.seg_sth:
                if args.supcon:           
                    masks = torch.cat([masks[0], masks[1]],dim=0) #2*bsz,256,256
                masks = masks.unsqueeze(1).float().cuda() #2*bsz,1,256,256    
                imgs = torch.cat([imgs, masks], dim=1)     # 2*bs, 2, 128,256,256           

            labels = labels.float().cuda()
            bsz = labels.shape[0]

            ## mixup
            if args.mixup:
                if args.supcon:
                    lam = 0.4
                    target_labels = torch.cat([labels, labels],dim=0)
                    mix_imgs, targets_a, targets_b, lam = mixup_data(imgs, target_labels, lam)

                    _, _, mix_logits = target_model(mix_imgs)
                    mix_logits1, mix_logits2 = torch.split(mix_logits, [bsz, bsz], dim=0) #bsz,n_classs
                    targets_a1, targets_a2 = torch.split(targets_a, [bsz, bsz], dim=0) #bsz,n_classs
                    targets_b1, targets_b2 = torch.split(targets_b, [bsz, bsz], dim=0) #bsz,n_classs


                    #mix_pred1 = F.softmax(mix_logits1)
                    #mix_pred2 = F.softmax(mix_logits2)
                    _, mix_predicted1 = torch.max(mix_logits1, dim=1)
                    _, mix_predicted2 = torch.max(mix_logits2, dim=1)

                    loss_func1 = mixup_criterion(targets_a1, targets_b1, lam)
                    loss_mixup1 = loss_func1(criterion_clf, mix_logits1)

                    loss_func2 = mixup_criterion(targets_a2, targets_b2, lam)
                    loss_mixup2 = loss_func2(criterion_clf, mix_logits2)
                    
                    loss_mix = 0.5*loss_mixup1+0.5*loss_mixup2
                else:
                    lam = 0.4
                    target_labels = labels
                    mix_imgs, targets_a, targets_b, lam = mixup_data(imgs, target_labels, lam)
                    # TODO 前三行修改
                    _,_, mix_logits = target_model(mix_imgs)
                    #mix_pred = F.softmax(mix_pred)
                    _, predicted = torch.max(mix_logits, dim=1)

                    loss_func = mixup_criterion(targets_a, targets_b, lam)
                    loss_mixup = loss_func(criterion_clf, mix_logits)

                    loss_mix = loss_mixup      

                    # pred_list.append(predicted.detach().cpu())
                    # label_list.append(labels.detach().cpu())

            # if not args.mixup: # mixup only
            _, features, logits = target_model(imgs) #2*bsz,128 #2*bsz,n_class
            if args.supcon:
                f1, f2 = torch.split(features, [bsz, bsz], dim=0) #bsz,128
                features = torch.cat([f1.unsqueeze(1),f2.unsqueeze(1)],dim=1) #bsz,2,128
                loss_con = criterion(features,labels)

                logits1, logits2 = torch.split(logits, [bsz, bsz], dim=0) #bsz,n_classs
                #pred1 = F.softmax(pred1)
                #pred2 = F.softmax(pred2)
                
                _, predicted1 = torch.max(logits1, dim=1)
                _, predicted2 = torch.max(logits2, dim=1)
                loss_clf = 0.5*criterion_clf(logits1,labels.long())+0.5*criterion_clf(logits2,labels.long())
                
                con_matx.add(predicted1.detach().cpu(),labels.detach().cpu())
                con_matx.add(predicted2.detach().cpu(),labels.detach().cpu())
                
                pred_list.append(predicted1.detach().cpu())
                label_list.append(labels.detach().cpu())
                pred_list.append(predicted2.detach().cpu())
                label_list.append(labels.detach().cpu())        

            else:
                #pred = F.softmax(pred)
                #con_matx.add(pred.detach().cpu(),labels.detach().cpu())
                #_, predicted = pred.max(1)
                loss_clf = criterion_clf(logits,labels.long())
                _, predicted = torch.max(logits, dim=1)
                con_matx.add(predicted.detach().cpu(),labels.detach().cpu())            
                pred_list.append(predicted.detach().cpu())
                label_list.append(labels.detach().cpu())
            if args.mixup and not args.supcon:
                loss = loss_mix #+ loss_clf
                loss_con = torch.zeros_like(loss)
                loss_clf = torch.zeros_like(loss)
            elif args.supcon and not args.mixup:
                loss = torch.exp(-s1)*loss_con+s1+torch.exp(-s2)*loss_clf+s2
                loss_mix = torch.zeros_like(loss)
            elif args.supcon and args.mixup:
                loss = torch.exp(-s1)*loss_con+s1+torch.exp(-s2)*(loss_mix + loss_clf)+s2
            else:
                loss = loss_clf
                loss_con = torch.zeros_like(loss)
                loss_mix = torch.zeros_like(loss)
            
            writer.add_scalar("Train/Loss_con_iter", loss_con.item(), epoch * len(train_loader) + i)
            writer.add_scalar("Train/Loss_clf_iter", loss_clf.item(), epoch * len(train_loader) + i)
            writer.add_scalar("Train/Loss_mix_iter", loss_mix.item(), epoch * len(train_loader) + i)


            total_loss1 += loss_con.item()
            total_loss2 += loss_clf.item()
            total_loss3 += loss_mix.item()

            if args.supcon:
                total += 2 * bsz
                correct += predicted1.eq(labels.long()).sum().item()
                correct += predicted2.eq(labels.long()).sum().item()
            else:
                total += bsz
                correct += predicted.eq(labels.long()).sum().item()                

            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr = scheduler.step()
            writer.add_scalar("Train/LR", lr, epoch * len(train_loader) + i)
                    
            pbar.set_description('loss_con: %.3f ' % (total_loss1 / (i+1))+ 'loss_clf: %.3f ' % (total_loss2 / (i+1)) + 'loss_mix: %.3f ' % (total_loss3 / (i+1)) +' acc: %.3f' % (correct/total))

        # recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
        # precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)

        lr = optimizer.param_groups[0]['lr']
        test_log.write('Epoch:%d  lr:%.5f  Loss_con:%.4f Loss_clf:%.4f Loss_mix:%.4f  acc:%.4f \n'%(epoch, lr, total_loss1 / count, total_loss2 / count, total_loss3 / count, correct/total))
        test_log.flush()
        
        writer.add_scalar("Train/Loss_con", total_loss1 / count, epoch)
        writer.add_scalar("Train/Loss_clf", total_loss2 / count, epoch)
        writer.add_scalar("Train/Loss_mix", total_loss3 / count, epoch)
        writer.add_scalar("Train/Accuracy", correct / total, epoch)

        if args.supcon:
            writer.add_scalar("Train/sigma1_weight", torch.exp(-s1).item(), epoch)
            writer.add_scalar("Train/sigma2_weight", torch.exp(-s2).item(), epoch)

        if (epoch + 1) % val_epoch == 0:
            val1(target_model,val_loader,epoch,test_log, optimizer, writer)
            if args.supcon:
                print(torch.exp(-s1).item(),torch.exp(-s2).item())
    writer.close()

@torch.no_grad()
def val1(net, val_loader, epoch,test_log, optimizer, writer):
    global best_f1
    net = net.eval()

    correct = .0
    total = .0
    con_matx = meter.ConfusionMeter(args.n_classes)
    pred_list = []
    label_list = []
    probs_list = []

    # total_ = []
    # label_ = []

    pbar = tqdm(val_loader, ascii=True)

    for i, (data, masks, label,id) in enumerate(pbar):
        data = data.unsqueeze(1)
        # data = data.repeat(1,3,1,1,1)   


        data = data.float().cuda()
        label = label.float().cuda()
        if args.seg_sth:
            masks = masks.unsqueeze(1)
            masks = masks.float().cuda()
            data = torch.cat([data, masks], dim=1)

        _, feat, logits = net(data)
        probs = F.softmax(logits, dim=1)
        probs_list.append(probs.detach().cpu())

        # print(feat.size())
        # total_.append(feat)
        # label_.append(label)

        #pred = F.softmax(pred)
        _, predicted = torch.max(logits, dim=1)

        pred_list.append(predicted.detach().cpu())
        label_list.append(label.detach().cpu())

        total += data.size(0)
        correct += predicted.eq(label.long()).sum().item()        
        con_matx.add(predicted.detach().cpu(),label.detach().cpu()) 
        pbar.set_description(' acc: %.3f'% (100.* correct / total))

    # ans = torch.cat(total_, dim=0)
    # ans = ans.cpu().numpy()

    # ans2 = torch.cat(label_, dim=0)
    # ans2 = ans2.cpu().numpy()
    # np.save('train_data.npy', ans)
    # np.save('train_label.npy', ans2)
    # print(ans.shape, ans2.shape)
    
    all_labels = torch.cat(label_list).cpu().numpy()
    all_probs = torch.cat(probs_list).cpu().numpy()
    try:
        y_true_bin = label_binarize(all_labels, classes=np.arange(args.n_classes))
        auc_macro = roc_auc_score(y_true_bin, all_probs, average='macro', multi_class='ovr')
    except Exception as e:
        print("ROC AUC calculation error:", e)
        auc_macro = float('nan')

    recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    f1 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average='macro')
    f1_4 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)

    
    print(correct, total)
    acc = 100.* correct/total

    writer.add_scalar("Val/Accuracy", acc, epoch)
    writer.add_scalar("Val/F1_macro", f1, epoch)
    writer.add_scalar("Val/Recall_macro", recall.mean(), epoch)
    writer.add_scalar("Val/Precision_macro", precision.mean(), epoch)
    writer.add_scalar("Val/AUC_macro", auc_macro, epoch)
    for i in range(len(recall)):
        writer.add_scalar(f"Val/Class_{i}_Recall", recall[i], epoch)
        writer.add_scalar(f"Val/Class_{i}_Precision", precision[i], epoch)
        writer.add_scalar(f"Val/Class_{i}_F1", f1_4[i], epoch)
    
    print('val epoch:', epoch, ' val acc: ', acc, 'recall:', recall, "precision:", precision, "f1_macro:",f1, 'f1:', f1_4, 'auc_macro:', auc_macro)
    print(str(con_matx.value()))
    test_log.write('Val Epoch:%d   Accuracy:%.4f   f1:%.4f   AUC:%.4f   con:%s \n'%(epoch, acc, f1, auc_macro, str(con_matx.value())))
    test_log.flush()
    
    cm = con_matx.value()
    fig = plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    writer.add_figure("Val/Confusion_Matrix", fig, epoch)
    plt.close(fig)

    if not args.val_certain_epoch:
        if f1 >= best_f1:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'f1': f1,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            save_name = os.path.join(save_dir, str(epoch) + '.pkl')
            torch.save(state, save_name)
            best_f1 = f1


if __name__ == "__main__":
    main()
        

