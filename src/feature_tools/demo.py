# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/19 23:46
@summary:
"""

import pandas as pd
import numpy as np
import featuretools as ft

clients = pd.read_csv('../data/feature-tools/clients.csv', parse_dates=['joined'])
loans = pd.read_csv('../data/feature-tools/loans.csv', parse_dates=['loan_start', 'loan_end'])
payments = pd.read_csv('../data/feature-tools/payments.csv', parse_dates=['payment_date'])
pd.set_option('display.max_colwidth', 200)
# print(clients.head())

#创建实体
es = ft.EntitySet(id='clients')

# 添加clients实体
es = es.entity_from_dataframe(entity_id='clients', dataframe=clients,
                              index='client_id', time_index='joined')

# 添加loads实体
es = es.entity_from_dataframe(entity_id='loans', dataframe=loans,
                              variable_types={'repaid': ft.variable_types.Categorical},
                              index='loan_id',
                              time_index='loan_start')


# 添加pyments实体
es = es.entity_from_dataframe(entity_id='payments',
                              dataframe=payments,
                              variable_types={'missed': ft.variable_types.Categorical},
                              make_index=True,
                              index='payment_id',
                              time_index='payment_date')

#添加实体关系
# 通过client_id 关联clients和loans实体
r_client_previous = ft.Relationship(es['clients']['client_id'],
                                    es['loans']['client_id'])
es = es.add_relationship(r_client_previous)

# 通过loan_id 关联payments和loans实体
r_payments = ft.Relationship(es['loans']['loan_id'],
                             es['payments']['loan_id'])
es = es.add_relationship(r_payments)


# 打印实体集
# print(es)

# 聚合特征,并生成新特征
features, feature_names = ft.dfs(entityset=es, target_entity='clients')
pd.set_option('display.max_columns', None)
pd.set_option('display.width',4000)
mapping = {"credit": 1, "home": 2, "cash": 3, "other": 4}
features['MODE(loans.loan_type)'] = features['MODE(loans.loan_type)'].map(mapping)
# print(type(features))
# print(np.array(features))
print(features.head(25))


# 聚合特征，通过指定聚合agg_primitives和转换trans_primitives生成新特征
features2, feature_names2 = ft.dfs(entityset=es, target_entity='clients',
                                 agg_primitives=['mean', 'max', 'percent_true', 'last'],
                                 trans_primitives=['years', 'month', 'subtract', 'divide'])
print(len(feature_names))