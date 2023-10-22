
import numpy as np
import pandas as pd

# 自作ライブラリインポート
import util
import importlib
importlib.reload(util) # ライブラリ更新時に対応

# データセット作成
def create_dataset(df_base, is_train):
    # 特徴量
    # 性別をベクトル化
    df_one = pd.get_dummies(df_base[["Sex"]], dummy_na=False, drop_first=False)
    df_one = df_one.astype(np.int64)

    x = pd.concat([df_one, df_base[["Pclass", "Fare"]]], axis=1)
    id = df_base[["PassengerId"]]
    if is_train:
        y = df_base[["Survived"]]
        return x, y, id
    else:
        return x, id

if __name__ == '__main__':

    # データ取得
    df_train = pd.read_csv("../../data/input/train.csv") # 学習データ
    df_test = pd.read_csv("../../data/input/test.csv")   # テストデータ

    # データセット作成
    x_train, y_train, id_train = create_dataset(df_train, True)
    x_test, id_test = create_dataset(df_test, False)

    # 最適化パラメータ探索
    # best_trial = util.optimize(util.objective_func, x_train, y_train)
    # print(best_trial.params)
    # print(util.params)

    # util.params.update(best_trial.params)

    # 仮パラメータ
    util.params = {
        'lambda_l1': 1.3406343673102123,
        'lambda_l2': 3.4482904089131434,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'num_leaves': 16,
        'n_estimators': 100000,
        'random_state': 123,
        'importance_type': 'gain',
    }

    # CV実行
    imp, metrics, model_list = util.train_cv(x_train, y_train, id_train, util.params, n_splits=5)

    # 予測

    # 結果を辞書に保存
    solution = {}
    
    # 各モデルで予測
    for i, model in enumerate(model_list):
        solution[str(i) + "_model"] = model.predict(x_test)

    # 辞書からDataFrameに変更
    solution = pd.DataFrame(solution)

    # 多数決 (最頻値)を取得
    solution_max = solution.mode(axis = 1).values

    # PassengerIdを取得
    PassengerId = np.array(df_test["PassengerId"]).astype(int)
    
    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(solution_max.astype(int), index = PassengerId, columns = ["Survived"])
    
    # my_tree_one.csvとして書き出し
    my_solution.to_csv("../../data/output/002/submission.csv", index_label = ["PassengerId"])