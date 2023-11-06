import fastcore
fastcore.__version__

import fastai2
fastai2.__version__

from fastai2.tabular.all import *
from IPython import get_ipython
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

if __name__ == '__main__':

    # データ取得
    df_train = pd.read_csv("../../data/input/train.csv") # 学習データ
    df_test = pd.read_csv("../../data/input/test.csv")   # テストデータ

    cat_names= [
            'Name', 'Sex', 'Ticket', 'Cabin', 
            'Embarked', 'Name_wiki', 'Hometown', 
            'Boarded', 'Destination', 'Lifeboat', 
            'Body'
    ]

    cont_names = [ 
        'PassengerId', 'Pclass', 'SibSp', 'Parch', 
        'Age', 'Fare', 'WikiId', 'Age_wiki','Class'
    ]

    splits = RandomSplitter(valid_pct=0.2)(range_of(df_train))

    to = TabularPandas(df_train, procs=[Categorify, FillMissing,Normalize],
                        cat_names = cat_names,
                        cont_names = cont_names,
                        y_names='Survived',
                        splits=splits)
    
    g_train =to.train.xs.columns.to_series().groupby(to.train.xs.dtypes).groups
    print(g_train)

    # xs'を使って、見慣れた方法でテーブルを取得する。Survived'カラムがないことに注意。
    to.train.xs

    # 目標値にも同じものを使っている
    to.train.ys.values.ravel()

    ### RANDOM FOREST
    X_train, y_train = to.train.xs, to.train.ys.values.ravel()
    X_valid, y_valid = to.valid.xs, to.valid.ys.values.ravel()

    X_train.head()