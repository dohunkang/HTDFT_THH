# A data-driven approach for the guided regulation of exposed facets in nanoparticles
Zihao Ye, Bo Shen, Dohun Kang, Jiahong Shen, Jin Huang, Zhe Wang, Liliang Huang, Christopher M. Wolverton, and Chad A. Mirkin

This set of supplemental materials contains the code and data associated with our paper titled "A data-driven approach for the guided regulation of exposed facets in nanoparticles".

## Repo Contents
- [data](./data): summarized data calculated from HT-DFT and Magpie feature set with preference
- [scripts](./scripts): python codes for summarize the overall HT-DFT calculation (HT_DFT_surfE.py) and analyze the relationship between host-guest atomic property and THH preference (machine_learning.py)


# System requirements
## Software Requirements
The codes are tested on Windows10 operating systems and python(v. 3.11.0) jupyter notebook.

# Installation Guide
## Package dependencies
The codes except for shap analysis are tested on following packages:
`pandas==2.0.0
seaborn==0.12.2
numpy==1.24.2
pickle==0.7.5
matplotlib==3.7.0
scikit-learn==1.2.2`

Shap analysis code is tested on following packages:
`pandas==2.0.2
seaborn==0.12.2
numpy==1.23.1
pickle==0.7.5
matplotlib==3.7.1
shap==0.41.0`

# Demo
## Scripts
The codes can be runned by python or python jupyter notebook with stated packages
`HT_DFT_surfE.py` creates 1) host-guest nanoparticle surface energy plot in Fig S5., 2) surface energy difference heatmap in Fig 1., and 3) preference heatmap in Fig S6.
`machine_learning.py` is used for feature importance analysis by forest classifier, shap analysis, and gaussian process classifier as shown in Fig 2.
