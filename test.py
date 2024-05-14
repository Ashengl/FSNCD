import argparse
import os

import loguru
from scipy.cluster import hierarchy
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits
from data.data_utils import CategoriesSampler, BigDatasetSampler
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights
import torch.nn as nn
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from collections import Counter
from tqdm import tqdm

from torch.nn import functional as F
from loguru import logger
from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path


# from scipy.optimize import minimize_scalar
# from functools import partial
# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def hierarchical_clustering(data, args):
    count = data.shape[0]
    threshold = args.threshold
    clusters = [[i] for i in range(data.shape[0])]
    cos_sim = torch.matmul(data, torch.transpose(data, -1, -2)) - 2 * torch.eye(data.shape[0]).to(device)
    cos_sim[:args.n_shot, :args.n_shot] = cos_sim[:args.n_shot, :args.n_shot]*1.1
    while 1:
        x, y = cos_sim.argmax()//data.shape[0], cos_sim.argmax()%data.shape[0]
        check_1 = [i for i in range(args.n_way) if i in clusters[x]]
        check_2 = [i for i in range(args.n_way) if i in clusters[y]]
        if len(check_1)>0 and len(check_2)>0:
            break
        tmp_feat = (data[x] * len(clusters[x]) + data[y] * len(clusters[y])) / (len(clusters[x]) + len(clusters[y]))
        data[x] = tmp_feat
        index = torch.arange(data.shape[0]).to(device)
        index = index!=y
        data = data[index]
        cos_sim = torch.matmul(data, torch.transpose(data, -1, -2)) - 2 * torch.eye(data.shape[0]).to(device)
        cos_sim = cos_sim
        clusters[x].extend(clusters[y])
        del clusters[y]

    feat_pro = data[[len(i) > threshold or len([j for j in range(args.n_way) if j in i]) != 0 for i in clusters]]
    pro_clu = [i for i in clusters if len(i) > threshold or len([j for j in range(args.n_way) if j in i]) != 0]
    feat_qur = data[[len(i) <= threshold and len([j for j in range(args.n_way) if j in i]) == 0 for i in clusters]]
    que_clu = [i for i in clusters if len(i) <= threshold and len([j for j in range(args.n_way) if j in i]) == 0]

    cos_sim = torch.matmul(feat_pro, torch.transpose(feat_qur, -1, -2))

    if len(pro_clu) == 0 or len(que_clu) == 0:
        if cos_sim.shape[0] == 0:
            print(1)
    if len(que_clu) > 0:
        aff = cos_sim.argmax(0)
        for i in range(len(que_clu)):
            pro_clu[aff[i]].extend(que_clu[i])


    pred = []
    for i in range(count):
        for j in range(len(pro_clu)):
            if i in pro_clu[j]:
                pred.append(j)
    return pred

def rankstate(data, args):
    cos_sim = torch.matmul(
        F.normalize(data, p=2, dim=-1),
        torch.transpose(F.normalize(data[:args.n_way], p=2, dim=-1), -1, -2)
    ) / args.temperature
    preds = []
    hash_dict = []
    for feat in cos_sim.topk(2).indices:
        if not feat in hash_dict:
            hash_dict.append(feat.cpu().tolist())
        preds.append(hash_dict.index(feat.cpu().tolist()))
    return preds


def loop_kmeans(data, args):
    kmeans = KMeans(n_clusters=args.n_way, random_state=0, n_init=20).fit(data)
    preds = kmeans.labels_
    last_center = kmeans.cluster_centers_
    count_all = Counter(preds)
    count_query = Counter(preds[args.n_way:])
    count_support = Counter(preds[:args.n_way])
    t = [count_query[j] // count_support[j] for j in range(args.n_way) if count_support[j] != 0 ]
    t.extend([count_query[j] for j in range(args.n_way) if count_support[j] == 0 ])
    t = np.array(t)[(np.array(t) != 0)].mean()*1.4
    while 1:
        center = []
        for j in range(len(set(preds))):
            n_clusters = count_support[j] if count_support[j] > 1 else 1 if count_query[j] <= t else 2
            if n_clusters == 1:
                tmp_cluster_centers_ = last_center[[j]]
            else:
                tmp_kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20).fit(data[preds==j])
                tmp_cluster_centers_= tmp_kmeans.cluster_centers_
            center.extend(tmp_cluster_centers_.tolist())
        kmeans = KMeans(n_clusters=len(center), init=np.array(center), n_init=1).fit(data)
        preds = kmeans.labels_
        last_center = kmeans.cluster_centers_
        count_query = Counter(preds[args.n_way:])
        count_support = Counter(preds[:args.n_way])
        t = [count_query[j] // count_support[j] for j in range(len(set(preds))) if count_support[j] != 0]
        t.extend([count_query[j] for j in range(len(set(preds))) if count_support[j] == 0])
        t = np.array(t)[(np.array(t) != 0)].mean()*args.alpha
        if len(list(set(preds[:args.n_way]))) >= args.n_way and max(count_query.values()) <= t:
            break
    return kmeans.labels_


def test_realtime(model, test_loader,
                epoch, save_name,
                args):

    model.eval()
    all_acc = []
    old_acc = []
    new_acc = []
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.to(device)
        mask = torch.zeros(images.shape[0])
        mask[: (args.n_shot + args.n_query)*args.n_way] = 1
        mask = mask.detach().cpu().numpy()
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        support_feat = feats[:args.n_way * args.n_shot].reshape(args.n_shot, args.n_way, -1)
        support_prototype = support_feat.mean(0)

        query_feat = feats[args.n_way * args.n_shot:]

        all_feats = torch.cat((support_prototype, query_feat), 0)
        data = F.normalize(all_feats, p=2, dim=-1)
        for i in range(data.shape[0]-args.n_way):
            tmp_data = data[[j for j in range(args.n_way)]+[i+args.n_way]]
            # cos_sim = torch.matmul(
            #     F.normalize(query_feat, p=2, dim=-1),
            #     torch.transpose(F.normalize(support_prototype, p=2, dim=-1), -1, -2)
            # ) / args.temperature
            # preds = cos_sim.argmax(-1)
            if args.method == 'ukc':
                preds = loop_kmeans(tmp_data.cpu().numpy(), args)
            elif args.method == 'shc':
                preds = hierarchical_clustering(tmp_data, args)
            elif args.method == 'rank':
                preds = rankstate(tmp_data, args)
            else:
                raise ValueError(f'Wrong method {args.method}, option:kmeans, shc, rank')

            query_targets = label[args.n_way * (args.n_shot-1):][[j for j in range(args.n_way)]+[i+args.n_way]]
            query_mask = mask[[i+args.n_way]]
            # kmeans = KMeans(n_clusters=args.n_way + args.n_nc, random_state=0, n_init='auto').fit(query_feat.cpu().numpy())
            # preds = kmeans.labels_

            all_acc_, old_acc_, new_acc_ = log_accs_from_preds(y_true=query_targets.cpu().numpy(), y_pred=np.array(preds),  mask=query_mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                        writer=args.writer, n_way=args.n_way)

            all_acc.append(all_acc_)
            old_acc.append(old_acc_)
            new_acc.append(new_acc_)
    # -----------------------
    # EVALUATE
    # -----------------------
    new_acc = np.array(new_acc)[np.logical_not(np.isnan(new_acc))]
    old_acc = np.array(old_acc)[np.logical_not(np.isnan(old_acc))]
    all_acc = np.array(all_acc)[np.logical_not(np.isnan(all_acc))]
    return all_acc.mean(), old_acc.mean(), new_acc.mean()


def test_normal(model, test_loader,
                epoch, save_name,
                args):

    model.eval()
    all_acc = []
    old_acc = []
    new_acc = []
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.to(device)
        mask = torch.zeros(images.shape[0])
        mask[: (args.n_shot + args.n_query)*args.n_way] = 1
        mask = mask.detach().cpu().numpy()
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        support_feat = feats[:args.n_way * args.n_shot].reshape(args.n_shot, args.n_way, -1)
        support_prototype = support_feat.mean(0)

        query_feat = feats[args.n_way * args.n_shot:]

        all_feats = torch.cat((support_prototype, query_feat), 0)
        data = F.normalize(all_feats, p=2, dim=-1)
        # cos_sim = torch.matmul(
        #     F.normalize(query_feat, p=2, dim=-1),
        #     torch.transpose(F.normalize(support_prototype, p=2, dim=-1), -1, -2)
        # ) / args.temperature
        # preds = cos_sim.argmax(-1)
        if args.method == 'ukc':
            preds = loop_kmeans(data.cpu().numpy(), args)
        elif args.method == 'shc':
            preds = hierarchical_clustering(data, args)
        elif args.method == 'rank':
            preds = rankstate(data, args)
        else:
            raise ValueError(f'Wrong method {args.method}, option:kmeans, shc, rank')

        query_targets = label[args.n_way * (args.n_shot-1):]
        query_mask = mask[args.n_way * args.n_shot:]
        # kmeans = KMeans(n_clusters=args.n_way + args.n_nc, random_state=0, n_init='auto').fit(query_feat.cpu().numpy())
        # preds = kmeans.labels_

        all_acc_, old_acc_, new_acc_ = log_accs_from_preds(y_true=query_targets.cpu().numpy(), y_pred=np.array(preds),  mask=query_mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer, n_way=args.n_way)
        all_acc.append(all_acc_)
        old_acc.append(old_acc_)
        new_acc.append(new_acc_)

    return sum(all_acc)/len(all_acc), sum(old_acc)/len(old_acc), sum(new_acc)/len(new_acc)


def test_large(model, test_loader,
                epoch, save_name,
                args):

    model.eval()
    all_acc = []
    old_acc = []
    new_acc = []
    all_feat = []
    all_mask = []
    all_label = []
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.to(device)
        mask = torch.zeros(images.shape[0])
        mask[: (args.n_shot + args.n_query)*args.n_way] = 1
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        all_feat.append(feats.cpu())
        all_mask.append(mask.cpu())
        all_label.append(label.cpu())
    feats = torch.cat(all_feat)
    mask = torch.cat(all_mask)
    label = torch.cat(all_label)
    data = F.normalize(feats, p=2, dim=-1)
    support_feat = data[:args.n_way * args.n_shot].reshape(args.n_shot, args.n_way, -1)
    support_prototype = support_feat.mean(0)
    query_feat = data[args.n_way * args.n_shot:(args.batchsize4large*2)]

    all_feats = torch.cat((support_prototype, query_feat), 0)
    remaining_feats = data[(args.batchsize4large*2):]
    # cos_sim = torch.matmul(
    #     F.normalize(query_feat, p=2, dim=-1),
    #     torch.transpose(F.normalize(support_prototype, p=2, dim=-1), -1, -2)
    # ) / args.temperature
    # preds = cos_sim.argmax(-1)
    if args.method == 'ukc':
        preds = loop_kmeans(all_feats.cpu().numpy(), args).tolist()
    elif args.method == 'shc':
        preds = hierarchical_clustering(all_feats, args)
    elif args.method == 'rank':
        preds = rankstate(all_feats, args)
    else:
        raise ValueError(f'Wrong method {args.method}, option:kmeans, shc, rank')

    prototype = []
    for i in range(max(preds)+1):
        prototype.append(all_feats[np.array(preds)==i].mean(0).unsqueeze(1))
    prototype = torch.cat(prototype, 1)
    cos_sim = torch.matmul(
        F.normalize(remaining_feats, p=2, dim=-1),
        F.normalize(prototype, p=2, dim=-1)
    ) / args.temperature
    remaining_preds = cos_sim.argmax(-1).detach().cpu().numpy()
    preds.extend(remaining_preds)
    query_targets = label[args.n_way * (args.n_shot-1):]
    query_mask = mask[args.n_way * args.n_shot:].detach().cpu().numpy()

    all_acc_, old_acc_, new_acc_ = log_accs_from_preds(y_true=query_targets.cpu().numpy(), y_pred=np.array(preds),  mask=query_mask,
                                                T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                writer=args.writer, n_way=args.n_way)

    all_acc.append(all_acc_)
    old_acc.append(old_acc_)
    new_acc.append(new_acc_)

    return sum(all_acc)/len(all_acc), sum(old_acc)/len(old_acc), sum(new_acc)/len(new_acc)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--n_shot', default=5, type=int)
    parser.add_argument('--n_nc', default=5, type=int)
    parser.add_argument('--n_query', default=15, type=int)
    parser.add_argument('--n_episode', default=50, type=int)  # episode num for each epoch

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--method', type=str, default='ukc', help='options: ukc, shc, rank')
    parser.add_argument('--alpha', type=float, default=1.4)
    parser.add_argument('--threshold', type=float, default=2.0)
    parser.add_argument('--batchsize4large', default=200, type=int)
    parser.add_argument('--task', type=str, default='realtime', help='options: normal, large, realtime')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    logger.add('log.txt')
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = f'/media/cs4007/adfc2692-0951-4a9b-8ea6-ebfa0e11323b/lcm/NCD/cache/metric_learn_gcd/log/{args.dataset_name}/checkpoints/model.pt'
        model = vits.__dict__['vit_base']()

        # model.half()

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            # model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False
    else:

        raise NotImplementedError

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    # train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, val_dataset = get_datasets(args.dataset_name, train_transform, test_transform, args)
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=128, shuffle=False)
    if args.task == 'normal':
        test_sampler = CategoriesSampler(
            test_dataset.label, num_episodes=500, const_loader=False,
            num_way=args.n_way, num_shot=args.n_shot, num_query=args.n_query, n_nc=args.n_nc
        )
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=test_sampler,
                                 pin_memory=True)
        all_acc_test, old_acc_test, new_acc_test = test_normal(model, test_loader,
                                                           epoch=0, save_name='Test ACC',
                                                           args=args)
    elif args.task == 'realtime':
        args.threshold = 0
        test_sampler = CategoriesSampler(
            test_dataset.label, num_episodes=500, const_loader=False,
            num_way=args.n_way, num_shot=args.n_shot, num_query=args.n_query, n_nc=args.n_nc
        )
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=test_sampler,
                                 pin_memory=True)
        all_acc_test, old_acc_test, new_acc_test = test_realtime(model, test_loader,
                                                           epoch=0, save_name='Test ACC',
                                                           args=args)
    elif args.task == 'large':
        test_sampler = BigDatasetSampler(
            test_dataset.label, num_episodes=250, const_loader=False,
            num_way=args.n_way, num_shot=args.n_shot, num_query=args.n_query, n_nc=args.n_nc, batch_size=args.batchsize4large
        )

        test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=test_sampler,
                                 pin_memory=True)
        all_acc_test, old_acc_test, new_acc_test = test_large(model, test_loader,
                                                           epoch=0, save_name='Test ACC',
                                                           args=args)
    else:
        raise ValueError(f'Wrong method {args.method}, option:normal, large, realtime')
        # all_.append(all_acc_test)
        # old_.append(old_acc_test)
        # new_.append(new_acc_test)
    logger.info(f'{args.method},{args.task},{args.n_nc}nc,{args.n_way}way,{args.n_query}query,{args.n_shot}shot,on{args.dataset_name},{args.alpha}')
    logger.info(f'{all_acc_test}, {old_acc_test}, {new_acc_test}')