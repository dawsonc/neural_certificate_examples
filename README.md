<div align="center">    
 
# Learning Neural Certificates

<!-- [![Conference](https://img.shields.io/badge/CoRL-Accepted-success)](https://openreview.net/forum?id=8K5kisAnb_p)
   
[![Arxiv](http://img.shields.io/badge/arxiv-eess.sy:2109.06697-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->

<!--  
Conference   
-->   
</div>
 
## Description
This repository contains example code for learning controllers with Lyapunov, Barrier, and Contraction certificates using neural networks. It is designed to accompany our survey article **TODO**.

## How to run
First, install dependencies   
```bash
# clone project   
git clone https://github.com/dawsonc/neural_certificate_examples

# install project   
cd neural_certificate_examples
conda create --name neural_certificate_examples python=3.9
conda activate neural_certificate_examples
pip install -e .   
pip install -r requirements.txt
```

Once installed, training examples can be run using e.g. `python neural_certificate_examples/training/train_clf.py`, and pre-trained models can be evaluated using the scripts in the `neural_certificate_examples/evaluation` directory.

### Citation
Until further notice, if you use this code in your own research, we ask that you cite our publication on this topic:
```
@article{dawson_neural_clbf_2021,
  title={Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier Functions},
  author={Charles Dawson, Zengyi Qin, Sicun Gao, Chuchu Fan},
  journal={5th Annual Conference on Robot Learning},
  year={2021}
}
```   
