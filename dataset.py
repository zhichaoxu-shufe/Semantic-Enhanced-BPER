import argparse
import numpy as np
import random
import sys
import time
import json
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader, Dataset


class SEBPERTrainDataset(Dataset):
	def __init__(self, args):
		self.trainset = np.load(os.path.join(args.input_dir, 'trainset.pickle'), allow_pickle=True)
		
		self.args = args
		self.ui_hist = self.get_hist()


	def get_hist(self):
		ui_hist = {}
		
		for i, row in enumerate(self.trainset):
			user_id = row[0]
			item_id = row[1]
			exp_id = row[2]
			if row[-1]:
				if (user_id,item_id) not in ui_hist.keys():
					ui_hist[(user_id, item_id)] = set()
				ui_hist[(user_id, item_id)].add(exp_id)

		return ui_hist

	def __getitem__(self, index):
		entry = self.trainset[index]
		user_id = entry[0]
		item_id = entry[1]
		exp_id = entry[2]
		neg_exp_id = entry[3]

		return {
		"user_id":user_id,
		"item_id":item_id,
		"exp_id":exp_id,
		"neg_exp_id":neg_exp_id,
		}

	def __len__(self):
		return self.trainset.shape[0]

	def collate_fn(self, batch):
		user_ids, item_ids, exp_ids, neg_exp_ids = [], [], [], []
		for entry in batch:
			user_ids.append(entry['user_id'])
			item_ids.append(entry['item_id'])
			exp_ids.append(entry['exp_id'])
			neg_exp_ids.append(entry['neg_exp_id'])

		return {
			'user_ids':torch.LongTensor([user_ids]),
			'item_ids':torch.LongTensor([item_ids]),
			'exp_ids':torch.LongTensor([exp_ids]),
			'neg_exp_ids':torch.LongTensor([neg_exp_ids])
		}

class SEBPERTestDataset(Dataset):
	def __init__(self, args):
		if args.test_mode == 'small':
			self.testset = np.load(os.path.join(args.input_dir, 'testset_small.pickle'), allow_pickle=True)
		elif args.test_mode == 'full':
			self.testset = np.load(os.path.join(args.input_dir, 'testset.pickle'), allow_pickle=True)

		self.args = args
		self.gt_dict()
		self.organize_testset()

	def organize_testset(self):
		self.testset = []
		for k,v in self.gt_dict.items():
			self.testset.append([k[0], k[1]])


	def __getitem__(self, index):
		entry = self.testset[index]
		user_id = entry[0]
		item_id = entry[1]
		return {
		"user_id":user_id,
		"item_id":item_id,
		}

	def __len__(self):
		return len(self.testset)

	def gt_dict(self):
		gt = {}
		for i, row in enumerate(self.testset):
			user = row[0]
			item = row[1]
			if (user, item) not in gt.keys():
				gt[(user, item)] = set()
			gt[(user, item)].add(row[2])
		self.gt_dict = {k: list(v) for k, v in gt.items()}
