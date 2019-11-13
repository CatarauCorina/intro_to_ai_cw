#!/usr/bin/env python3

import io
import os
import torch
import torchvision
import pandas as pd
import PIL
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from operator import itemgetter
import pickle


class CelebAImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(CelebAImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]

        tuple_with_path = {
            'image': original_tuple[0],
            'target': original_tuple[1],
            'path': (path.rsplit('/', 1)[-1],)
        }
        list_img = [original_tuple[0], original_tuple[1], path.rsplit('/', 1)[-1]]
        tuple_with_path = original_tuple + (path.rsplit('/', 1)[-1],)
        return list_img


class CelebADataLoader(Dataset):

    def __init__(self, dataset_type='gt15',
                 root_dir='../one_shot_learning/data/img_alig_split/',
                 gt15_dir=os.path.join(os.getcwd(),'..\\one_shot_learning\\data\\img_alig_split_gt_15\\img_alig_split_gt_15\\')):
        if dataset_type == 'gt15':
            self.root_dir = gt15_dir
        else:
            self.root_dir = root_dir

        self.dataset, self.train_loader = self.load_data(self.root_dir)
        self.load_attributes_files()
        self.load_labels_names_and_files()
        self.merge_data()
        self.data_paths = self.data_labels['path']
        self.identity_classes = list(pd.Series(self.dataset.targets).unique())
        self.transform = None
        self.split_train_test()
        return

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_paths[idx]
        image = io.imread(img_name)

        sample = {'image': image, 'label': self.data_labels['label'][idx]}

        return sample

    def get_images_with_features(self, features):
        condition = None
        for feature in features:
            if condition is None:
                condition = (self.img_attr[feature] == 1)
            else:
                condition = (self.img_attr[feature] == 1) & condition
        list_img_conditions = list(
            self.img_attr[condition]['img_name']
        )
        images_features_filters = [img for i, img in enumerate(self.dataset)
                                   if img[2] in list_img_conditions]
        return images_features_filters

    def load_data(self, dir_imgs):
        data_path = dir_imgs
        transform = transforms.Compose(
            [transforms.ToTensor()])
        train_dataset = CelebAImageFolderWithPaths(
            root=data_path,
            transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True
        )

        return train_dataset, train_loader

    def load_data_class(self, target, dataset='train'):
        ds_targets = []
        ds = None
        if dataset == 'train':
            ds = self.train_dataset
            ds_targets = self.train_targets
        elif dataset == 'test':
            ds = self.test_dataset
            ds_targets = self.test_targets
        targets = torch.tensor(ds_targets)
        target_idx = (targets == target).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx)
        loader = torch.utils.data.DataLoader(ds, sampler=sampler)
        return loader

    def get_data_subset(self, idx, dataset='train'):
        if dataset == 'train':
            ds = self.train_dataset
        elif dataset == 'test':
            ds = self.test_dataset
        targets = torch.tensor(idx)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(targets)
        loader = torch.utils.data.DataLoader(ds, sampler=sampler)
        return loader

    def load_attributes_files(self, file_name='data/list_attr_celeba_processed.csv'):
        self.img_attr = pd.read_csv(os.path.join(os.getcwd(),'..\\one_shot_learning\\data\\list_attr_celeba_processed.csv'))
        return self.img_attr

    def load_labels_names_and_files(self, name=os.path.join(os.getcwd(),'..\\one_shot_learning\\data\\data_labels_names.csv')):
        self.data_labels = pd.read_csv(name)
        return self.data_labels

    def merge_data(self):
        df1 = self.img_attr
        df2 = self.data_labels
        self.merged = df1.merge(df2, left_on='img_name', right_on='img_name')
        return

    def split_train_test(self,
                         split_size=0.8,
                         shuffle_dataset=True,
                         random_seed=42):
        #dataloader_shuffle = torch.utils.data.DataLoader(self.dataset, shuffle=shuffle_dataset)
        dataset_size = len(self.dataset)
        train_size = int(split_size * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size]
        )
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        all_targets = self.dataset.targets
        train_indices = train_dataset.indices
        test_indices = test_dataset.indices
        self.train_targets = list(itemgetter(*train_indices)(all_targets))
        self.test_targets = list(itemgetter(*test_indices)(all_targets))
        return train_dataset, test_dataset


class SiameseDatasetCreator:

    def __init__(self):
        self.celeb_loader = CelebADataLoader()

    def filter_targets(self, type_ds):
        sample_classes = None
        if type_ds == 'train':
            sample_classes = pd.Series(self.celeb_loader.train_targets)
        elif type_ds == 'test':
            sample_classes = pd.Series(self.celeb_loader.test_targets)
        return sample_classes

    def get_image_by_class(self, class_val, type_ds='train'):
        sample_classes = self.filter_targets(type_ds)
        filtered = sample_classes[sample_classes == class_val]
        idx_filtered = filtered.index
        data_filtered = self.celeb_loader.get_data_subset(idx_filtered, type_ds)
        return data_filtered, len(filtered), idx_filtered

    def get_images_excluding_class(self, class_true, type_ds='train'):
        sample_classes = self.filter_targets(type_ds)
        filtered = sample_classes[sample_classes != class_true]
        idx_filtered = filtered.index
        data_filtered = self.celeb_loader.get_data_subset(idx_filtered, type_ds)
        return data_filtered, len(filtered), idx_filtered

    def create_pair_siamese(self, class_true, type_ds='train'):
        class_1_data, nr_samples_class, filtered_idx = self.get_image_by_class(
            class_true, type_ds
        )
        targets = np.zeros((nr_samples_class,))
        targets[(nr_samples_class // 2):] = 1
        class_not_1_data, nr_samples_class_not_1, filtered_idx_not_1 = self.get_images_excluding_class(
            class_true, type_ds
        )
        nr_channels, img_height, img_width = tuple(self.celeb_loader.dataset[0][0].shape)
        pairs = [np.zeros((nr_samples_class, img_height, img_width, nr_channels)) for i in range(2)]

        for i in range(nr_samples_class):
            id_net_1 = np.random.choice(filtered_idx)
            img_class_1 = list(self.celeb_loader.get_data_subset([[id_net_1]], type_ds))[0][0].reshape(
                img_height, img_width, nr_channels
            )
            pairs[0][i, :, :, :] = img_class_1

            id_net_2_same = np.random.choice(filtered_idx)
            img_class_1_2 = list(self.celeb_loader.get_data_subset([[id_net_2_same]], type_ds))[0][0].reshape(
                img_height, img_width, nr_channels
            )

            id_net_2_diff = np.random.choice(filtered_idx_not_1)
            img_class_2 = list(self.celeb_loader.get_data_subset([[id_net_2_diff]], type_ds))[0][0].reshape(
                img_height, img_width, nr_channels
            )

            if i >= nr_samples_class // 2:
                pairs[1][i, :, :, :] = img_class_2
            else:
                pairs[1][i, :, :, :] = img_class_1_2
        return pairs, targets

    # batch-size defines the number of pairs we need
    def generate_verification_input(self, nr_ex_class=10, batch_size=10, type_ds='train'):
        if type_ds == 'train':
            sample_classes = pd.Series(self.celeb_loader.train_targets).unique()
            ds = self.celeb_loader.train_dataset
        elif type_ds == 'test':
            sample_classes = pd.Series(self.celeb_loader.test_targets).unique()
            ds = self.celeb_loader.test_dataset

        for class_identity in sample_classes:
            pairs_class, target_class = self.create_pair_siamese(class_identity, type_ds)
            yield pairs_class, target_class

def main():
    siamese_in_creator = SiameseDatasetCreator()
    nr_channels, height, width = siamese_in_creator.celeb_loader.dataset[0][0].shape
    print(len(siamese_in_creator.celeb_loader.dataset))
    train_siamese_data = siamese_in_creator.generate_verification_input(type_ds='train')
    print(next(train_siamese_data))
    return

if __name__ == '__main__':
    main()


