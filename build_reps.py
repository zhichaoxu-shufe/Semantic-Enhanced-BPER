import numpy as np
import pickle
import json
import os
import sys
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	args = parser.parse_args()

	reps = np.load(os.path.join(args.input_dir, 'trained_doc_reps.pickle'), allow_pickle=True)
	doc_vec = np.random.rand(len(reps), 768)
	for i, row in enumerate(reps):
		doc_vec[row[0]]=row[1]

	testset = np.load(os.path.join(args.input_dir, 'trainset.pickle'), allow_pickle=True)

	user_vec, item_vec = {}, {}

	for i, row in enumerate(testset):
		if row[0] not in user_vec.keys():
			user_vec[row[0]] = doc_vec[row[3]].mean(axis=0)
		if row[1] not in item_vec.keys():
			item_vec[row[1]] = doc_vec[row[4]].mean(axis=0)

	doc_dict = {}
	for i, row in enumerate(doc_vec):
		doc_dict[i] = row

	with open(os.path.join(args.input_dir, 'user_rep.pkl'), 'wb') as f:
		pickle.dump(user_vec, f)

	with open(os.path.join(args.input_dir, 'item_rep.pkl'), 'wb') as f:
		pickle.dump(item_vec, f)

	with open(os.path.join(args.input_dir, 'doc_rep.pkl'), 'wb') as f:
		pickle.dump(doc_dict, f)