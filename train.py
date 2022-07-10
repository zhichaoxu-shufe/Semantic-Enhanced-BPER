import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from torch.utils.data.distributed import DistributedSampler

from dataset import *
from engine import *

import argparse
import numpy as np
import random
import sys
import os
import json
import copy
from tqdm import tqdm
import pickle

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)

	# amazon: 109121, 47113, 33767
	# tripadvisor: 123374, 200475, 76293
	# yelp: 895729, 164779, 126696

	# # tripadvisor
	# parser.add_argument('--num_users', type=int, default=123374)
	# parser.add_argument('--num_items', type=int, default=200475)
	# parser.add_argument('--num_exps', type=int, default=76293)

	# # amazon
	# parser.add_argument('--num_users', type=int, default=109121)
	# parser.add_argument('--num_items', type=int, default=47113)
	# parser.add_argument('--num_exps', type=int, default=33767)

	# yelp
	parser.add_argument('--num_users', type=int, default=895729)
	parser.add_argument('--num_items', type=int, default=164779)
	parser.add_argument('--num_exps', type=int, default=126696)

	parser.add_argument('--optimizer', type=str, default='adam')
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--weight_decay', type=float, default=1e-4)
	parser.add_argument('--use_cuda', type=int, default=1)

	parser.add_argument('--batch_size', type=int, default=32)

	parser.add_argument('--mu', type=float, default=0.7)
	parser.add_argument('--clip', type=float, default=5.0)
	
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--test_per_epoch', type=int, default=10)

	parser.add_argument('--test_only', type=int, default=0)
	parser.add_argument('--test_mode', type=str, default='small')
	parser.add_argument('--save_ckpt', type=int, default=0)
	parser.add_argument('--load_model_from_checkpoint', type=int, default=0)
	parser.add_argument('--designated_ckpt', type=str)

	parser.add_argument('--random_init', type=int, default=0)
	parser.add_argument('--partial_init', type=int, default=0)
	parser.add_argument('--latent_dim', type=int, default=768)


	args = parser.parse_args()
	trainset = SEBPERTrainDataset(args)

	testset = SEBPERTestDataset(args)
	
	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate_fn)
	test_loader = DataLoader(testset, batch_size=1, shuffle=False)
	
	engine = SEBPEREngine(args)


	if not args.test_only:
		for epoch in range(1, args.epochs+1):
			epoch_loss = engine.train_an_epoch(train_loader,0)
			print('epoch loss: ', epoch_loss)
			if epoch % args.test_per_epoch == 0:
				ranklist = engine.output_ranklist(test_loader)
				engine.evaluate(testset.gt_dict, ranklist)
			if args.save_ckpt:
				save_checkpoint(engine.model, args.input_dir, 'sebper_epoch_{}.pt'.format(epoch))

	if args.test_only:
		if args.load_model_from_checkpoint:
			engine.model = load_checkpoint(engine.model, args.designated_ckpt)

		ranklist = engine.output_ranklist(test_loader)
		engine.evaluate(testset.gt_dict, ranklist)