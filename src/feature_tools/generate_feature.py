# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import featuretools as ft


def get_data():
    clients = pd.read_csv('../data/feature-tools/clients.csv', parse_dates=['joined'])
    loans = pd.read_csv('../data/feature-tools/loans.csv', parse_dates=['loan_start', 'loan_end'])
    payments = pd.read_csv('../data/feature-tools/payments.csv', parse_dates=['payment_date'])
    return clients,loans,payments

def get_es():
    """创建实体及实体关系"""
    clients, loans, payments = get_data()
    # 创建实体
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

    # 添加实体关系
    # 通过client_id 关联clients和loans实体
    r_client_previous = ft.Relationship(es['clients']['client_id'],
                                        es['loans']['client_id'])
    es = es.add_relationship(r_client_previous)

    # 通过loan_id 关联payments和loans实体
    r_payments = ft.Relationship(es['loans']['loan_id'],
                                 es['payments']['loan_id'])
    es = es.add_relationship(r_payments)
    return es

def convert():
    """转换获取特征"""
    es = get_es()
    features, feature_names = ft.dfs(entityset=es, target_entity='clients')
    mapping = {"credit": 1, "home": 2, "cash": 3, "other": 4}
    features['MODE(loans.loan_type)'] = features['MODE(loans.loan_type)'].map(mapping)
    return features, feature_names

def get_together():
    """聚合特征，通过指定聚合agg_primitives和转换trans_primitives生成新特征"""
    es = get_es()
    features, feature_names = ft.dfs(entityset=es, target_entity='clients',
                                       agg_primitives=['mean', 'max', 'percent_true', 'last'],
                                       trans_primitives=['years', 'month', 'subtract', 'divide'])
    return features, feature_names