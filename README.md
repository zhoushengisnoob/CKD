# Collaborative Knowledge Distillation for Heterogeneous Information Network Embedding (CKD)
Code for The Web Conference 2022 Paper "Collaborative Knowledge Distillation for Heterogeneous Information Network Embedding" [[Paper]](https://zhoushengisnoob.github.io/papers/WWW2022.pdf)

## Dataset
We provide six datasets used in this paper, three of which are only used for node classification(nc), and the other three are used for both node classification and link prediction(lp).
In this repo, we provide the ACM dataset, the rest data can be found in [GoogleDrive](https://drive.google.com/drive/folders/1dOmetBd4wVUClUHtqYrA-r3eiXxYH8B-?usp=sharing).

* ACM(nc)
* ACM2(nc and lp)
* DBLP(nc)
* DBLP2(nc and lp)
* Freebase(nc)
* Freebase(nc and lp)

## Input Data
ACM、DBLP、Freebase contains 5 files

```
1. info.dat: it contains the description information of nodes, links and labels.
...

2. label.dat: each line contains node id, node text description, node category and node label.
...

3. label.dat.test: label set used for testing.
...

4. link.dat: each line contains node id, node id, link type and link weight.
...

5. node: each line contains node id, node text description, node category and node feature.
...
```

ACM2、DBLP2、PubMed contains 6 files, in addition to the above five files, it also contains link.dat.test


## How to use
We have provided the **Pytorch**  implementation of CKD.
The requirements of the running environment is listed in **requirements.txt**.

You can create the environment with anaconda: 

    conda install --yes --file requirements.txt

or virtualenv:

    pip install -r requirements.txt

### Transform
Before training, the dataset should be transformed first. You can view the conversion details in the readme.md file under folder Transform, then, the code can be run by:

    bash dblp_transform.sh (for DBLP)

or

    bash transform.sh (for other datasets)


### Run
Before running, switch to the ckd / Model / CKD / src directory, then, the code can be run by:

    python3 main_for_freebase.py (for Freebase)

or

    python3 main.py (for other datasets)


### Evaluate
After the training, run evaluate.sh in the Evaluate folder to get NC or LP results
