import torch
import torch.nn.functional as F

import sys
import pickle
import os
import time

class SIMF(torch.nn.Module):
	def __init__(self, args):
		super(SIMF, self).__init__()
		self.args = args

		self.embedding_user = torch.nn.Embedding(num_embeddings=self.args.num_users, embedding_dim=args.latent_dim)
		self.embedding_item = torch.nn.Embedding(num_embeddings=self.args.num_items, embedding_dim=args.latent_dim)
		self.embedding_exp_u = torch.nn.Embedding(num_embeddings=self.args.num_exps, embedding_dim=args.latent_dim)
		self.embedding_exp_i = torch.nn.Embedding(num_embeddings=self.args.num_exps, embedding_dim=args.latent_dim)

		self.user_bias = torch.nn.Embedding(num_embeddings=self.args.num_exps, embedding_dim=1)
		self.item_bias = torch.nn.Embedding(num_embeddings=self.args.num_exps, embedding_dim=1)

		print('model initialization completed')

		if args.random_init:
			# torch.nn.init.uniform_(self.user_bias.weight, -0.01, 0.01)
			# torch.nn.init.uniform_(self.item_bias.weight, -0.01, 0.01)
			self.item_bias.weight.data = torch.zeros_like(self.item_bias.weight.data)
			self.user_bias.weight.data = torch.zeros_like(self.user_bias.weight.data)
			torch.nn.init.uniform_(self.embedding_user.weight, -0.01, 0.01)
			torch.nn.init.uniform_(self.embedding_item.weight, -0.01, 0.01)
			torch.nn.init.uniform_(self.embedding_exp_u.weight, -0.01, 0.01)
			torch.nn.init.uniform_(self.embedding_exp_i.weight, -0.01, 0.01)

		if not args.random_init:
			self.init_weight()
			self.embedding_user.weight.data /= 100
			self.embedding_item.weight.data /= 100
			self.embedding_exp_u.weight.data /= 100
			self.embedding_exp_i.weight.data /= 100
			torch.nn.init.uniform_(self.user_bias.weight, -0.01, 0.01)
			torch.nn.init.uniform_(self.item_bias.weight, -0.01, 0.01)

		# self.softmax = torch.nn.Softmax(dim=0)
		self.logsigmoid = torch.nn.LogSigmoid()
		

		print('embedding initialization completed')

	def init_weight(self):
		if not self.args.partial_init:
			self.embedding_item.weight.data=torch.zeros_like(self.embedding_item.weight.data)
			self.embedding_user.weight.data=torch.zeros_like(self.embedding_user.weight.data)
			with open(os.path.join(self.args.input_dir, 'user_rep.pkl'), 'rb') as f:
				user_dict = pickle.load(f)
			with torch.no_grad():
				for k,v in user_dict.items():
					self.embedding_user.weight.data[k]+=torch.nn.Parameter(torch.tensor(v))

			del user_dict

			with open(os.path.join(self.args.input_dir, 'item_rep.pkl'), 'rb') as f:
				item_dict = pickle.load(f)
			with torch.no_grad():
				for k,v in item_dict.items():
					self.embedding_item.weight.data[k]+=torch.nn.Parameter(torch.tensor(v))
			del item_dict

		self.embedding_exp_u.weight.data=torch.zeros_like(self.embedding_exp_u.weight.data)
		self.embedding_exp_i.weight.data=torch.zeros_like(self.embedding_exp_i.weight.data)

		with open(os.path.join(self.args.input_dir, 'doc_rep.pkl'), 'rb') as f:
			doc_dict = pickle.load(f)

		with torch.no_grad():
			for k,v in doc_dict.items():
				self.embedding_exp_u.weight.data[k]+=torch.nn.Parameter(torch.tensor(v))
				self.embedding_exp_i.weight.data[k]+=torch.nn.Parameter(torch.tensor(v))
		del doc_dict

		# print(self.embedding_exp.weight.requires_grad, self.embedding_user.weight.requires_grad, self.embedding_item.weight.requires_grad)


	def forward(self, user_indices, item_indices, exp_indices, neg_exp_indices):
		user_embedding=self.embedding_user(user_indices) # [bz, latent_dim]
		item_embedding=self.embedding_item(item_indices)

		u_exp_embedding=self.embedding_exp_u(exp_indices)
		i_exp_embedding=self.embedding_exp_i(exp_indices)

		u_neg_exp_embedding=self.embedding_exp_u(neg_exp_indices)
		i_neg_exp_embedding=self.embedding_exp_i(neg_exp_indices)

		# print(user_embedding.shape, item_embedding.shape, exp_embedding.shape)

		user_exp_product = torch.mul(user_embedding, u_exp_embedding).sum(axis=1)
		user_score = self.user_bias(exp_indices).squeeze()+user_exp_product
		user_neg_exp_product = torch.mul(user_embedding, u_neg_exp_embedding).sum(axis=1)
		user_neg_score = self.user_bias(neg_exp_indices).squeeze()+user_neg_exp_product

		item_exp_product = torch.mul(item_embedding, i_exp_embedding).sum(axis=1)
		item_score = self.item_bias(exp_indices).squeeze()+item_exp_product
		item_neg_exp_product = torch.mul(item_embedding, i_neg_exp_embedding).sum(axis=1)
		item_neg_score = self.item_bias(neg_exp_indices).squeeze()+item_neg_exp_product

		user_score = user_score - user_neg_score
		item_score = item_score - item_neg_score
		loss = torch.cat((user_score, item_score), dim=0)
		# loss = self.logsigmoid(loss).mean(axis=0)
		loss = self.logsigmoid(loss)

		# user_score = torch.mul(user_embedding, u_exp_embedding)+self.user_bias(exp_indices)
		# user_neg_score = torch.mul(user_embedding, u_neg_exp_embedding)+self.user_bias(neg_exp_indices)
		# user_score = user_score.sum(axis=1)-user_neg_score.sum(axis=1)

		# item_score = torch.mul(item_embedding, i_exp_embedding)+self.item_bias(exp_indices)
		# item_neg_score = torch.mul(item_embedding, i_neg_exp_embedding)+self.item_bias(neg_exp_indices)
		# item_score = item_score.sum(axis=1)-item_neg_score.sum(axis=1)

		# loss = torch.cat((user_score, item_score), dim=0)
		# loss = self.logsigmoid(loss).mean(axis=0)

		return -loss

	def compute(self, user_indice, item_indice, exp_indice):
		user_embedding=self.embedding_user(user_indice) # [bz, latent_dim]
		item_embedding=self.embedding_item(item_indice)
		u_exp_embedding=self.embedding_exp_u(exp_indice)
		i_exp_embedding=self.embedding_exp_i(exp_indice)

		# print(user_embedding.shape, item_embedding.shape, exp_embedding.shape)

		user_exp_product = torch.mul(user_embedding, u_exp_embedding).sum(axis=1)
		user_score = self.user_bias(exp_indice).squeeze()+user_exp_product

		item_exp_product = torch.mul(item_embedding, i_exp_embedding).sum(axis=1)
		item_score = self.item_bias(exp_indice).squeeze()+item_exp_product

		score = self.args.mu*user_score + (1-self.args.mu)*item_score
		return score.item()