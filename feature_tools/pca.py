# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from feature_tools import generate_feature

def pca():
    """使用PCA进行降维"""
    features, feature_names = generate_feature.convert()
    ipca = PCA(n_components=10, copy=True)
    data = ipca.fit_transform(features)
    # da = ipca.inverse_transform(data)
    print(ipca.score_samples(features))

pca()
