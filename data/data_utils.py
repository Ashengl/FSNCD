import numpy as np
from torch.utils.data import Dataset
import torch

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)



class CategoriesSampler:

    def __init__(self, labels, num_episodes, num_way, num_shot, num_query, const_loader, n_nc):
        self.num_way = num_way
        self.n_nc = n_nc
        self.num_shot = num_shot
        self.num_query = num_query
        self.const_loader = const_loader
        self.num_episodes = num_episodes
        self.m_ind = []
        self.batches = []

        labels = np.array(labels)
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            if not ind.shape[0] == 0:
                ind = torch.from_numpy(ind)
                self.m_ind.append(ind)

        # for c in classes:
        #     l = self.m_ind[c.item()]
        #     pos = torch.randperm(l.size()[0])
        #     batch_gallery.append(l[pos[: self.num_shot + self.num_query]])
        #     batch_query.append(l[pos[self.num_shot: self.num_shot + self.num_query]])
        #
        # # batch = torch.cat(batch_gallery + batch_query)
        # batch_gallery = torch.cat(batch_gallery).reshape(self.num_way, self.num_shot).T.reshape(-1)
        # batch_query = torch.cat(batch_query).reshape(self.num_way, self.num_query).T.reshape(-1)
        # batch = torch.cat((batch_gallery, batch_query))
        # self.batches.append(batch)

        if self.const_loader:
            for i_batch in range(self.num_episodes):
                batch = []
                classes = torch.randperm(len(self.m_ind))[:self.num_way]
                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch.append(l[pos[: self.num_shot + self.num_query]])

                batch = torch.stack(batch).t().reshape(-1)
                self.batches.append(batch)

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        if not self.const_loader:
            for batch_idx in range(self.num_episodes):
                batch = []
                new_batch = []
                classes = torch.randperm(len(self.m_ind))[: (self.num_way + self.n_nc)]
                new_class = classes[self.num_way:]
                classes = classes[: self.num_way]

                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch.append(l[pos[: self.num_shot + self.num_query]])
                for c in new_class:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    new_batch.append(l[pos[: self.num_query]])

                batch = torch.stack(batch).t().reshape(-1)
                new_batch = torch.stack(new_batch).t().reshape(-1)
                yield torch.cat((batch, new_batch), 0)
        else:
            for batch_idx in range(self.num_episodes):
                # batch = torch.stack(self.batches[i_batch]).reshape(-1)
                yield self.batches[batch_idx]

class TrainSampler:

    def __init__(self, labels, num_episodes, num_way, num_shot, num_query, const_loader, n_nc, batch_size):
        self.num_way = num_way
        self.batch_size = batch_size
        self.n_nc = n_nc
        self.num_shot = num_shot
        self.num_query = num_query
        self.const_loader = const_loader
        self.num_episodes = num_episodes
        self.m_ind = []
        self.batches = []

        labels = np.array(labels)
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            if not ind.shape[0] == 0:
                ind = torch.from_numpy(ind)
                self.m_ind.append(ind)

        if self.const_loader:
            for i_batch in range(self.num_episodes):
                batch = []
                classes = torch.randperm(len(self.m_ind))[:self.num_way]
                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch.append(l[pos[: self.num_shot + self.num_query]])

                batch = torch.stack(batch).t().reshape(-1)
                self.batches.append(batch)

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        if not self.const_loader:
            for batch_idx in range(self.num_episodes):
                iter_batch = []
                for i in range(self.batch_size):
                    batch = []
                    new_batch = []
                    classes = torch.randperm(len(self.m_ind))[: (self.num_way + self.n_nc)]
                    new_class = classes[self.num_way:]
                    classes = classes[: self.num_way]

                    for c in classes:
                        l = self.m_ind[c.item()]
                        pos = torch.randperm(l.size()[0])
                        batch.append(l[pos[: self.num_shot + self.num_query]])
                    for c in new_class:
                        l = self.m_ind[c.item()]
                        pos = torch.randperm(l.size()[0])
                        new_batch.append(l[pos[: self.num_query]])

                    batch = torch.stack(batch).t().reshape(-1)
                    new_batch = torch.stack(new_batch).t().reshape(-1)
                    iter_batch.append(torch.cat((batch, new_batch), 0))
                yield torch.stack(iter_batch).reshape(-1)
        else:
            for batch_idx in range(self.num_episodes):
                # batch = torch.stack(self.batches[i_batch]).reshape(-1)
                yield self.batches[batch_idx]

class BigDatasetSampler:
    def __init__(self, labels, num_episodes, num_way, num_shot, num_query, const_loader, n_nc, batch_size=200):
        self.num_way = num_way
        self.n_nc = n_nc
        self.num_shot = num_shot
        self.num_query = num_query
        self.const_loader = const_loader
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.m_ind = []
        self.batches = []

        labels = np.array(labels)
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            if not ind.shape[0] == 0:
                ind = torch.from_numpy(ind)
                self.m_ind.append(ind)

        if self.const_loader:
            for i_batch in range(self.num_episodes):
                batch = []
                classes = torch.randperm(len(self.m_ind))[:self.num_way]
                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch.append(l[pos[: self.num_shot + self.num_query]])

                batch = torch.stack(batch).t().reshape(-1)
                self.batches.append(batch)
        assert batch_size%(self.num_way + self.n_nc) == 0
        self.classes = torch.randperm(len(self.m_ind))[: (self.num_way + self.n_nc)]
        self.batch = batch_size//(self.num_way + self.n_nc)
        self.num_episodes = np.array([self.m_ind[i].shape[0] for i in self.classes]).min()//self.batch

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        if not self.const_loader:
            for batch_idx in range(self.num_episodes):
                batch = []
                new_batch = []
                new_class = self.classes[self.num_way:]
                classes = self.classes[: self.num_way]

                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.arange(l.size()[0])
                    batch.append(l[pos[self.batch * batch_idx : self.batch * (batch_idx+1)]])
                for c in new_class:
                    l = self.m_ind[c.item()]
                    pos = torch.arange(l.size()[0])
                    new_batch.append(l[pos[self.batch * batch_idx : self.batch * (batch_idx+1)]])

                batch = torch.stack(batch).t().reshape(-1)
                new_batch = torch.stack(new_batch).t().reshape(-1)
                yield torch.cat((batch, new_batch), 0)
        else:
            for batch_idx in range(self.num_episodes):
                # batch = torch.stack(self.batches[i_batch]).reshape(-1)
                yield self.batches[batch_idx]