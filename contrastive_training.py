import argparse
import os
from scipy.cluster import hierarchy
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits
from data.data_utils import CategoriesSampler, TrainSampler
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights
import torch.nn as nn
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from loguru import logger
from tqdm import tqdm

from torch.nn import functional as F

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, args):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def train(projection_head, model, train_loader, test_loader, args):

    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss()
    best_test_acc_lab = 0

    for epoch in range(args.epochs):

        loss_record = AverageMeter()

        projection_head.train()
        model.train()
        tbar = tqdm(train_loader)
        for batch_idx, batch in enumerate(tbar):

            images, class_labels, uq_idxs = batch
            images, class_labels = images.to(device), class_labels.to(device)
            # Extract features with base model
            features = model(images)
            # Pass features through projection head
            features = projection_head(features)


            # L2-normalize features
            features = torch.nn.functional.normalize(features, dim=-1)

            sup_con_loss = sup_con_crit(features.unsqueeze(1), labels=class_labels)


            # Total loss
            loss = sup_con_loss

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tbar.set_postfix_str(f" loss: {sup_con_loss.item()}")

        logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        if epoch > 0.8*args.epochs:
            with torch.no_grad():
                logger.info('Testing on unlabelled examples in the training data...')
                # all_acc, old_acc, new_acc = test_cluster(model, train_loader,
                #                                         epoch=epoch, save_name='Train ACC Unlabelled',
                #                                         args=args)
                # (0.8992000000000004 0.8864000000000005 0.9162666666666671)

                logger.info('Testing on disjoint test set...')
                all_acc_test, old_acc_test, new_acc_test = test_cluster(model, test_loader,
                                                                       epoch=epoch, save_name='Test ACC',
                                                                       args=args)
                # (0.7640000000000002 0.7627999999999993 0.7656000000000003)
                # logger.info(all_acc_test)
                # logger.info(old_acc_test)
                # logger.info(new_acc_test)
                # logger.info('Train Acc All: {:.4f}, Old: {:.4f}, New: {:.4f}'.format(all_acc, old_acc, new_acc))
                logger.info('Test Acc All: {:.4f}, Old: {:.4f}, New: {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
            # # LOG
            # # ----------------
            # args.writer.add_scalar('Loss', loss_record.avg, epoch)
            # args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
            # args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)
            #
            # print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
            #                                                                       new_acc))
            # print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
            #                                                                         new_acc_test))
            #
            # # Step schedule
            # exp_lr_scheduler.step()
            #
            # torch.save(model.state_dict(), args.model_path)
            # print("model saved to {}.".format(args.model_path))
            #
            # torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
            # print("projection head saved to {}.".format(args.model_path[:-3] + '_proj_head.pt'))
            #
            # if old_acc_test > best_test_acc_lab:
            #
            #     print(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
            #     print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
            #                                                                           new_acc))
            #
            #     torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
            #     print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
            #
            #     torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
            #     print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best.pt'))
            #
            #     best_test_acc_lab = old_acc_test


def hierarchical_clustering(data, args):
    count = data.shape[0]
    threshold = 2
    clusters = [[i] for i in range(data.shape[0])]
    cos_sim = torch.matmul(data, torch.transpose(data, -1, -2)) - 2 * torch.eye(data.shape[0]).to(device)
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
        clusters[x].extend(clusters[y])
        del clusters[y]

    feat_pro = data[[len(i) > threshold or len([j for j in range(args.n_way) if j in i]) != 0 for i in clusters]]
    pro_clu = [i for i in clusters if len(i) > threshold or len([j for j in range(args.n_way) if j in i]) != 0]
    feat_qur = data[[len(i) <= threshold and len([j for j in range(args.n_way) if j in i]) == 0 for i in clusters]]
    que_clu = [i for i in clusters if len(i) <= threshold and len([j for j in range(args.n_way) if j in i]) == 0]

    cos_sim = torch.matmul(feat_pro, torch.transpose(feat_qur, -1, -2))

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

def test_cluster(model, test_loader,
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
        targets = label

        support_feat = feats[:args.n_way * args.n_shot].reshape(args.n_shot, args.n_way, -1)
        support_targets = label[:args.n_way * args.n_shot]
        support_prototype = support_feat.mean(0)

        query_feat = feats[args.n_way * args.n_shot:]

        all_feats = torch.cat((support_prototype, query_feat), 0)
        data = F.normalize(all_feats, p=2, dim=-1)
        if args.method == 'kmeans':
            for i in range(data.shape[0]):
                kmeans = KMeans(n_clusters=i+1, random_state=0, n_init=20).fit(data.cpu().numpy())
                preds = kmeans.labels_
                if len(list(set(preds[:5]))) >= args.n_way:
                    break
        elif args.method == 'hier':
            preds = hierarchical_clustering(data, args)

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
    # -----------------------
    # EVALUATE
    # -----------------------


    return sum(all_acc)/len(all_acc), sum(old_acc)/len(old_acc), sum(new_acc)/len(new_acc)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='imagenet_100', help='options: cifar10, cifar100, scars')
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--n_shot', default=5, type=int)
    parser.add_argument('--n_nc', default=5, type=int)
    parser.add_argument('--n_query', default=15, type=int)
    parser.add_argument('--n_episode', default=50, type=int)  # episode num for each epoch
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--devices', default=None, type=str)
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--method', default='kmeans', type=str)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device("cuda:2")
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'])
    logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    logger.info(args)
    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path

        model = vits.__dict__['vit_base']()

        # model.half()

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

        if args.warmup_model_dir is not None:
            logger.info(f'Loading weights from {args.warmup_model_dir}')
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

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    elif args.base_model == 'vit_ibot':

        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = '/home/ubuntu/data/lcm/Pretraining/checkpoint_teacher.pth'

        model = vits.__dict__['vit_base']()

        # model.half()

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'], strict=True)

        if args.warmup_model_dir is not None:
            logger.info(f'Loading weights from {args.warmup_model_dir}')
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

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
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

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------

    train_sampler = CategoriesSampler(
        train_dataset.label, num_episodes=args.n_episode, const_loader=False,
        num_way=args.n_way, num_shot=args.n_shot, num_query=args.n_query, n_nc=args.n_nc
    )
    test_sampler = CategoriesSampler(
        test_dataset.label, num_episodes=200, const_loader=False,
        num_way=args.n_way, num_shot=args.n_shot, num_query=args.n_query, n_nc=args.n_nc
    )
    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,  batch_sampler=test_sampler, pin_memory=True)
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=128, shuffle=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projection_head.to(device)

    ### parallelism

    if args.devices is not None:
        logger.info(args.devices)
        device_list = [int(x) for x in args.devices.split(',')]
        model = nn.DataParallel(model, device_ids=device_list)
        projection_head = nn.DataParallel(projection_head, device_ids=device_list)
    # ----------------------
    # TRAIN
    # ----------------------
    train(projection_head, model, train_loader, test_loader, args)