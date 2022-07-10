import random
import argparse
import numpy as np
import pandas as pd
import random, sys, time, json
import os
from tqdm import tqdm

import torch
# from transformers import AlbertTokenizer

random.seed(42)


def read_ids_id2exp(input_dir):
	ids = np.load(os.path.join(input_dir, 'IDs.pickle'), allow_pickle=True)
	with open(os.path.join(input_dir, 'id2exp.json'), 'r') as f:
		id2exp = json.load(f)

	return {'ids': ids, 'id2exp': id2exp}

def name2id(ids):
	user2id, item2id = {}, {}
	user_count, item_count = 0, 0
	for entry in ids:
		if entry['user'] not in user2id.keys():
			user2id[entry['user']] = user_count
			user_count += 1
		if entry['item'] not in item2id.keys():
			item2id[entry['item']] = item_count
			item_count += 1
	return user2id, item2id

def read_train_test_index(input_dir):
	with open(os.path.join(input_dir, 'train.index'), 'r') as f:
		train_index_str = f.read()
	train_index_str = train_index_str.split(' ')
	train_index = []
	for i, index in enumerate(train_index_str):
		train_index.append(int(index))
	with open(os.path.join(input_dir, 'test.index'), 'r') as f:
		test_index_str = f.read()
	test_index_str = test_index_str.split(' ')
	test_index = []
	for i, index in enumerate(test_index_str):
		test_index.append(int(index))
	return train_index, test_index

def build_id(input_dir, args):
	ids, id2exp = read_ids_id2exp(input_dir)['ids'], read_ids_id2exp(input_dir)['id2exp']
	user2id, item2id = name2id(ids)

	user2id_json = {k: str(v) for k, v in user2id.items()}
	item2id_json = {k: str(v) for k, v in item2id.items()}

	with open(os.path.join(input_dir, 'user2id.json'), 'w') as fp:
		json.dump(user2id_json, fp)
	with open(os.path.join(input_dir, 'item2id.json'), 'w') as fp:
		json.dump(item2id_json, fp)

def process_query(input_dict, args):
	from spacy.lang.en import English
	import spacy
	import string
	import re
	spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
	nlp = English()
	def remove_stopwords(tokenized_text, stop_words):
		lst = []
		for token in tokenized_text:
			if token.lower() not in stop_words:
				lst.append(token.lower())
		return ' '.join(lst)

	punctuation = string.punctuation
	punctuation = re.sub("'", "", punctuation)
	punctuation = re.sub("-", "", punctuation)
	def remove_punctuation(sentence, punctuation):
		return " ".join("".join([" " if ch in punctuation else ch for ch in sentence]).split())

	exp2id_processed = {}
	tokenizer=nlp.Defaults.create_tokenizer(nlp)
	for k,v in input_dict.items():
		sentence = remove_punctuation(k, punctuation)
		if args.remove_stopwords:
			tokens = sentence.split(' ')
			processed = remove_stopwords(tokens, spacy_stopwords)
			exp2id_processed[processed] = v
		else:
			exp2id_processed[sentence.lower()]=v

	return exp2id_processed


def rebuild_exp_id(input_dir, args):
	ids, id2exp = read_ids_id2exp(input_dir)['ids'], read_ids_id2exp(input_dir)['id2exp']
	train_index, test_index = read_train_test_index(input_dir)
	
	exp2id = {}
	id_map = {}
	count = 0

	for i, row in enumerate(ids):
		exp_idx = row['exp_idx']
		for j in exp_idx:
			if id2exp[j] not in exp2id.keys():
				exp2id[id2exp[j]] = count
				id_map[j] = count
				count += 1

	exp2id_json = {k: str(v) for k, v in exp2id.items()}
	if args.process_query:
		exp2id_json = process_query(exp2id_json, args)


	id2exp_json_new = {v:k for k,v in exp2id_json.items()}
	id_map_json = {k: str(v) for k, v in id_map.items()}

	with open(os.path.join(input_dir, 'exp2id.json'), 'w') as fp:
		json.dump(exp2id_json, fp)
	with open(os.path.join(input_dir, 'id_map.json'), 'w') as fp:
		json.dump(id_map_json, fp)
	with open(os.path.join(input_dir, 'id2exp_new.json'), 'w') as fp:
		json.dump(id2exp_json_new, fp)


def build_train_test(input_dir, num_negs):
	# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

	ids, id2exp = read_ids_id2exp(input_dir)['ids'], read_ids_id2exp(input_dir)['id2exp']
	train_index, test_index = read_train_test_index(input_dir)

	with open(os.path.join(input_dir, 'id_map.json'), 'r') as f:
		id_map_json = json.load(f)
	with open(os.path.join(input_dir, 'exp2id.json'), 'r') as f:
		exp2id_json = json.load(f)

	id_map = {k: int(v) for k, v in id_map_json.items()}
	user2id, item2id = name2id(ids)

	# build user hist and item hist
	user_hist, item_hist = {}, {}
	for i, idx in enumerate(train_index):
		if user2id[ids[idx]['user']] not in user_hist.keys():
			user_hist[user2id[ids[idx]['user']]]=set()
		for exp_idx in ids[idx]['exp_idx']:
			user_hist[user2id[ids[idx]['user']]].add(id_map[exp_idx])
		if item2id[ids[idx]['item']] not in item_hist.keys():
			item_hist[item2id[ids[idx]['item']]]=set()
		for exp_idx in ids[idx]['exp_idx']:
			item_hist[item2id[ids[idx]['item']]].add(id_map[exp_idx])

	user_doc, item_doc = {}, {}
	id2exp = {int(v): k for k, v in exp2id_json.items()}

	exp_pool = set(id2exp.keys())

	for k,v in user_hist.items():
		for i, exp_id in enumerate(v):
			if i==0:
				user_doc[k]=id2exp[exp_id]
			else:
				user_doc[k]+='.'
				user_doc[k]+=id2exp[exp_id]
	for k,v in item_hist.items():
		for i, exp_id in enumerate(v):
			if i==0:
				item_doc[k]=id2exp[exp_id]
			else:
				item_doc[k]+='.'
				item_doc[k]+=id2exp[exp_id]
	
	trainset = []
	testset = []

	random.shuffle(test_index)

	pbar=tqdm(total=len(test_index))
	for i, idx in enumerate(test_index):
		pbar.update(1)
		if i == 10000:
			break
		for j, exp_idx in enumerate(ids[idx]['exp_idx']):
			entry=[]
			user_id = user2id[ids[idx]['user']]
			item_id = item2id[ids[idx]['item']]
			exp_id = id_map[exp_idx]
			user_doc = list(user_hist[user_id])
			item_doc = list(item_hist[item_id])
			entry.append(user_id)
			entry.append(item_id)
			entry.append(exp_id)
			entry.append(user_doc)
			entry.append(item_doc)
			testset.append(entry)
	pbar.close()

	testset=np.array(testset)
	testset.dump(os.path.join(input_dir, 'testset_small.pickle'))

	if not args.testset_only:
		pbar=tqdm(total=len(train_index))
		for i, idx in enumerate(train_index):
			pbar.update(1)
			for j, exp_idx in enumerate(ids[idx]['exp_idx']):
				user_id = user2id[ids[idx]['user']]
				item_id = item2id[ids[idx]['item']]
				exp_id = id_map[exp_idx]

				user_doc = list(user_hist[user_id])
				item_doc = list(item_hist[item_id])

				neg_exp_ids = list(random.sample(exp_pool-set([exp_id]), num_negs))
				for k in neg_exp_ids:
					entry = [user_id, item_id, exp_id, k, user_doc, item_doc]
					trainset.append(entry)
		pbar.close()
		trainset=np.array(trainset)
		trainset.dump(os.path.join(input_dir, 'trainset.pickle'))
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--model', type=str, default='albert')
	parser.add_argument('--num_negs', type=int, default=3)
	parser.add_argument('--process_query', type=int, default=0)
	parser.add_argument('--remove_stopwords', type=int, default=0)

	parser.add_argument('--testset_only', type=int, default=0)


	args = parser.parse_args()

	build_id(args.input_dir, args)
	rebuild_exp_id(args.input_dir, args)
	build_train_test(args.input_dir, args.num_negs)
