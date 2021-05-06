#!/usr/bin/env bash
WORKDIR=$HOME/workspace

mkdir -p ${WORKDIR}/input/shopee-product-matching/
cd ${WORKDIR}/input/shopee-product-matching/
kaggle competitions download -c shopee-product-matching
unzip shopee-product-matching.zip
rm shopee-product-matching.zip

mkdir -p ${WORKDIR}/input/shopeeproductmatchingrequirements
cd ${WORKDIR}/input/shopeeproductmatchingrequirements
kaggle datasets download -d ar90ngas/shopeeproductmatchingrequirements
unzip shopeeproductmatchingrequirements.zip
rm shopeeproductmatchingrequirements.zip

cd $WORKDIR
git clone https://ar90n:${GITHUB_TOKEN}@github.com/ar90n/shopee-product-matching.git
cd shopee-product-matching
pip3 install -r requirements-dev.txt
pip3 install torch==1.8.1+cu111 torchtext==0.9.1 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install git+https://github.com/rwightman/pytorch-image-models
pip3 install -e .
