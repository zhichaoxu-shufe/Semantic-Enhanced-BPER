import torch
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict

from dataset import *
from sebper import *
from evaluate import *

import argparse
import numpy as np
import random
import sys
import time
import os
import json
import copy
import logging
from tqdm import tqdm


def use_optimizer(network, args):
	if args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(
			network.parameters(), 
			lr=args.lr, 
			weight_decay=args.weight_decay,
			)
	elif args.optimizer == 'asgd':
		optimizer = torch.optim.ASGD(
			network.parameters(),
			lr=args.lr,
			weight_decay=args.weight_decay,
			)
	elif args.optimizer == 'adam':
		optimizer = torch.optim.Adam(
			network.parameters(),
			lr=args.lr,
		)

	return optimizer

def check_grad_norm(model):
	total_norm = 0
	for p in model.parameters():
		param_norm = p.grad.detach().data.norm(2)
		total_norm += param_norm.item()**2
	print(total_norm)

def save_checkpoint(model, output_dir, model_name):
	ckpt_dir = os.path.join(output_dir, 'checkpoints')
	if not os.path.isdir(ckpt_dir):
		os.mkdir(ckpt_dir)
	torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{model_name}.pt'))

def load_checkpoint(model, checkpoint_path):
	return model.load_state_dict(torch.load(checkpoint_path))


class Engine(object):
	def __init__(self, args):
		self.args = args
		self.opt = use_optimizer(self.model, args)


	def train_an_epoch(self,  train_loader, epoch_id):
		if self.args.use_cuda:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = 'cpu'
		self.model.to(self.device)

		self.model.train()
		epoch_loss = 0

		for batch_id, batch in tqdm(enumerate(train_loader)):
			self.opt.zero_grad()

			user_ids = batch['user_ids'].squeeze(dim=0).to(self.device)
			item_ids = batch['item_ids'].squeeze(dim=0).to(self.device)
			exp_ids = batch['exp_ids'].squeeze(dim=0).to(self.device)
			neg_exp_ids = batch['neg_exp_ids'].squeeze(dim=0).to(self.device)

			batch_loss = self.model.forward(user_ids, item_ids, exp_ids, neg_exp_ids).mean(axis=0)

			batch_loss.backward()
			# check_grad_norm(self.model)
			# torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
			self.opt.step()
			epoch_loss += batch_loss.detach().item()

		return epoch_loss/len(train_loader)

	def output_ranklist(self, test_loader):
		if self.args.use_cuda:
			self.model.to('cpu')
		self.device = 'cpu'
		self.model.eval()

		user_exp_score_dict = {}
		item_exp_score_dict = {}

		self.user_bias_array = self.model.user_bias.weight.data.squeeze().numpy()
		self.item_bias_array = self.model.item_bias.weight.data.squeeze().numpy()
		self.user_embed_matrix = self.model.embedding_user.weight.data.numpy()
		self.item_embed_matrix = self.model.embedding_item.weight.data.numpy()
		self.u_exp_embed_matrix = self.model.embedding_exp_u.weight.data.numpy()
		self.i_exp_embed_matrix = self.model.embedding_exp_i.weight.data.numpy()

		exp_pool = [i for i in range(self.args.num_exps)]
		ranklist = {}

		with torch.no_grad():
			ranklist = {}
			for i, batch in tqdm(enumerate(test_loader),disable=False):
				user_id = batch['user_id'].item()
				item_id = batch['item_id'].item()
				user_ids = [user_id]*len(exp_pool)
				item_ids = [item_id]*len(exp_pool)

				scores = self.batchify(user_ids, item_ids, exp_pool)
				sorted_scores_index = scores.argsort()[::-1][:100].tolist()
				ranklist[(user_id, item_id)]=sorted_scores_index

		return ranklist

	def batch_mul(self, user_block, item_block, exp_block):
		user_embed=self.user_embed_matrix[user_block]
		item_embed=self.item_embed_matrix[item_block]
		u_exp_embed=self.u_exp_embed_matrix[exp_block]
		i_exp_embed=self.i_exp_embed_matrix[exp_block]
		user_bias=self.user_bias_array[exp_block]
		item_bias=self.item_bias_array[exp_block]

		scores=self.args.mu*(np.multiply(user_embed, u_exp_embed).sum(axis=1)+user_bias)+\
			(1-self.args.mu)*(np.multiply(item_embed, i_exp_embed).sum(axis=1)+item_bias)
		return scores


	def batchify(self, user_list, item_list, exp_list, block_size=32):
		step = len(user_list)//block_size+1
		scores = np.zeros(len(user_list))
		for i in range(step):
			user_block=user_list[block_size*i:block_size*(i+1)]
			item_block=item_list[block_size*i:block_size*(i+1)]
			exp_block=exp_list[block_size*i:block_size*(i+1)]
			scores[block_size*i:block_size*(i+1)]=self.batch_mul(user_block,item_block,exp_block)
		return scores

	def evaluate(self, gt, ranklist, cut_off=10):
		recall, precision, ndcg, hit, reciprocal_rank = print_metrics_with_rank_cutoff(ranklist, gt, cut_off)
		return recall, precision, ndcg, hit


class SEBPEREngine(Engine):
	def __init__(self, args):
		self.model = SEBPER(args)
		super(SEBPEREngine, self).__init__(args)

# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--input_dir', type=str, required=True)
# 	parser.add_argument('--num_users', type=int, default=109121)
# 	parser.add_argument('--num_items', type=int, default=47113)
# 	parser.add_argument('--num_exps', type=int, default=33767)

# 	parser.add_argument('--optimizer', type=str, default='sgd')
# 	parser.add_argument('--lr', type=float, default=1e-4)
# 	parser.add_argument('--use_cuda', type=int, default=1)

# 	parser.add_argument('--batch_size', type=int, default=256)

# 	parser.add_argument('--mu', type=float, default=0.7)


# 	args = parser.parse_args()
# 	trainset = SIMFTrainDataset(args)

# 	testset = SIMFTestDataset(args)
	
# 	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, collate_fn=trainset.collate_fn)
# 	test_loader = DataLoader(testset, batch_size=1, shuffle=False)
	
# 	engine=SIMFEngine(args)

# 	engine.output_ranklist(test_loader)

# 	engine.train_an_epoch(train_loader,0)