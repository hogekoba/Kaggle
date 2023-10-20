
import numpy as np
import pandas as pd
import os
import pickle
import gc 

# 分布確認
import ydata_profiling as pdp

# 可視化
import matplotlib.pyplot as plt

# 前処理
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

# バリデーション
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# 評価指標
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# モデリング: lightgbm
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")


# matplotilbで日本語表示したい場合はこれをinstallしてインポートする
import japanize_matplotlib

import seaborn as sns # Samuel Norman Seabornからとっている

# 分析の関数化
def check_croostab(x, hue, data, df_train):
    sns.countplot(x = x, hue=hue, data=data)
    plt.title(x + "による生存者数")
    plt.show()
    display(pd.crosstab(df_train[x], df_train["Survived"], normalize="index"))

import optuna

# 探索しないハイパーパラメータ
params_base = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    'n_estimators': 100000,
    "bagging_freq": 1,
    "seed": 123,
}

# LightGBMのハイパーパラメータ
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary', 
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 16,
    'n_estimators': 100000,
    "random_state": 123,
    "importance_type": "gain",
}

# 目的関数
def objective(trial):
    # 探索するハイパーパラメータ
    params_tuning = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-5, 1e-2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 1e2, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 1e2, log=True),
    }
    params_tuning.update(params_base)
    
    # モデル学習・評価
    list_metrics = []
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(x_train, y_train))
    for nfold in np.arange(5):
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :]
        x_va, y_va = x_train.loc[idx_va, :], y_train.loc[idx_va, :]
        model = lgb.LGBMClassifier(**params_tuning)
        model.fit(x_tr,
                  y_tr,
                  eval_set=[(x_tr,y_tr), (x_va,y_va)],
                  early_stopping_rounds=100,
                  verbose=0,
                 )
        y_va_pred = model.predict_proba(x_va)[:,1]
        metric_va = accuracy_score(y_va, np.where(y_va_pred>=0.5, 1, 0))
        list_metrics.append(metric_va)
    
    # 評価値の計算
    metrics = np.mean(list_metrics)
    
    return metrics

# 最適化実行
def optimize(objective):
    sampler = optuna.samplers.TPESampler(seed=123)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=30)

    trial = study.best_trial
    print("acc(best)={:.4f}".format(trial.value))
    display(trial.params)


def train_cv(input_x,
             input_y,
             input_id,
             params,
             n_splits=5,
            ):
    
    # 結果格納用
    metrics = []
    imp = pd.DataFrame()
    model_list = []

    # K分割検証法で学習用と検証用に分ける
    cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(input_x, input_y))

    # ループ回数分 LightGBMを試す
    for nfold in np.arange(n_splits):
        # 区切り線
        print("-"*20, nfold, "-"*20)

        # 学習データ、検証データのインデックスを取得
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        
        # インデックスのデータを取得
        x_tr, y_tr = input_x.loc[idx_tr, :], input_y.loc[idx_tr, :]
        x_va, y_va = input_x.loc[idx_va, :], input_y.loc[idx_va, :]
        print("x_train", x_tr.shape, "y_valid", y_tr.shape)
        print("x_valid", x_va.shape, "y_valid", y_va.shape)

        # Yデータの偏り確認
        print("y_train:{:.3f}, y_tr:{:.3f}, y_va:{:.3f}".format(
            input_y["Survived"].mean(),
            y_tr["Survived"].mean(),
            y_va["Survived"].mean(),
        ))

        # LightGBMモデル作成
        model = lgb.LGBMClassifier(**params)
        model.fit(x_tr,
                  y_tr,
                  eval_set=[(x_tr,y_tr), (x_va,y_va)],
                  early_stopping_rounds=100,
                  verbose=100,
                 )
        model_list.append(model)

        # 推論
        y_tr_pred = model.predict(x_tr)
        y_va_pred = model.predict(x_va)
        
        # 正解と予測から正解率を算出
        metric_tr = accuracy_score(y_tr, y_tr_pred)
        metric_va = accuracy_score(y_va, y_va_pred)
        print("[accuracy] tr: {:.2f}, va: {:.2f}".format(metric_tr, metric_va))    

        # 全体結果に格納
        metrics.append([nfold, metric_tr, metric_va])
        
        # 重要度の記録
        _imp = pd.DataFrame({"col":input_x.columns, "imp":model.feature_importances_, "nfold":nfold})
        imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

    # まとめ結果を表示
    print("-"*20, "result", "-"*20)
    metrics = np.array(metrics)
    print(metrics)

    # 正確性の平均、偏差
    print("[cv ] tr: {:.2f}+-{:.2f}, va: {:.2f}+-{:.2f}".format(
        metrics[:,1].mean(), metrics[:,1].std(),
        metrics[:,2].mean(), metrics[:,2].std(),
    ))

    # 重要度の平均偏差
    imp = imp.groupby("col")["imp"].agg(["mean", "std"])
    imp.columns = ["imp", "imp_std"]
    imp = imp.reset_index(drop=False)

    print("Done.")
    
    return imp, metrics, model_list