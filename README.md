<div align="center">
 
## ORCHID: ORal Cancer Histology Image Dataset

[![Project](http://img.shields.io/badge/Project%20Page-3d3d8f.svg)](https://ai-orchid.github.io/ORCHID-web/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.1001.2234-B31B1B.svg)](#)
<!--[![Demo](http://img.shields.io/badge/Demo-9acbff.svg)](https://huggingface.co/spaces/lukemelas/deep-spectral-segmentation)
[![Conference](http://img.shields.io/badge/CVPR-2022-4b44ce.svg)](#)-->

</div>

### Description
This code accompanies the ORCHID (ORal Cancer Histology Image Dataset) and contains information on the technical validation of the dataset.

### Abstract

Oral cancer is a global health challenge with a difficult histopathological diagnosis. The accurate histopathological interpretation of oral cancer tissue samples remains difficult. However, early diagnosis is very challenging due to a lack of experienced pathologists and inter-observer variability in diagnosis. The application of artificial intelligence (deep learning algorithms) for oral cancer histology images is very promising for rapid diagnosis. However, it requires a quality annotated dataset to build AI models. We present ORCHID (ORal Cancer Histology Image Database), a specialized database generated to advance research in AI-based histology image analytics of oral cancer and precancer. The ORCHID database is an extensive multicenter collection of high resolution images captured at 1000X magnification(100X objective lens), encapsulating various oral cancer and precancer categories, such as oral submucous fibrosis (OSMF) and oral squamous cell carcinoma (OSCC). Additionally, it also contains grade-level sub-classifications for OSCC, such as well-differentiated (WD), moderately-differentiated (MD), and poorly-differentiated (PD). Furthermore, the database seeks to bolster the creation and validation of innovative artificial intelligence-based rapid diagnostics for OSMF and OSCC, along with subtypes.

### How to run   

#### Dependencies
The minimal set of dependencies is listed in `requirements-ORCHID.txt` please ensure you create a conda environment with ```Python 3.12.*```. Running this code on CPU is extensively time consuming, and hence we have employed the use of an Nvidia GeForce RTX 4090 GPU. In order to install GPU-related dependencies, along with the requirements, we have created the ```initialise.sh``` script that you can run (After activating your conda environment to install all required dependencies).

#### Data
You must place the entirety of the Data in a location on your machine storage and provide the absolute path of this data to the ```train-normal-osmf-oscc.sh```, and the ```train-wdoscc-mdoscc-pdoscc.sh``` files. The data must be split into train(70%), val(20%), and test(10%) sets and placed as shown below:
```
|Dataset
  |-normal
  |-osmf
  |-wdoscc
  |-mdoscc
  |-pdoscc
```
(Note: the data is supplied in as compressed tar files, they must be uncompressed before being placed in the above format.)

## Licenses

### Dataset License
The dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). For more details, see the [LICENSE](LICENSE) file.

### Source Code License
The source code is licensed under the MIT License. For more details, see the [LICENSE_SOURCE_CODE](LICENSE_SOURCE_CODE) file.



