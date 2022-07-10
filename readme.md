## Learning to Rank Rationales for Explainable Recommendation
This is the official code implementation for arXiv manuscript: [Learning to Rank Rationales for Explainable Recommendation](https://arxiv.org/pdf/2206.05368.pdf)

#### Citation:
```bibtex
@article{xu2022learning,
  title={Learning to Rank Rationales for Explainable Recommendation},
  author={Xu, Zhichao and Han, Yi and Yang, Tao and Tran, Anh and Ai, Qingyao},
  journal={arXiv preprint arXiv:2206.05368},
  year={2022}
}
```

#### Dependency
```bash
pip3 install -r requirements.txt
```

#### Download the EXTRA dataset from [Link](https://lifehkbueduhk-my.sharepoint.com/personal/16484134_life_hkbu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F16484134%5Flife%5Fhkbu%5Fedu%5Fhk%2FDocuments%2FSIGIR21%2DEXTRA%2DDatasets&ga=1)
put IDs.pickle, id2exp.json, train.index and test.index under same folder

#### Preprocess
```bash
python preprocess.py --input_dir your_input_dir
```

#### Build user, item and doc representations
```bash
python build_reps.py --input_dir your_input_dir
```

#### Train SE-BPER
```bash
python train.py --input_dir your_input_dir
```
