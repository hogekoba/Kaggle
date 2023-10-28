
import numpy as np
import pandas as pd
import seaborn as sns # Samuel Norman Seabornからとっている

# 自作ライブラリインポート
import util
import importlib
importlib.reload(util) # ライブラリ更新時に対応

# データセット作成
def create_dataset(df, feature_list):
    # 推定に使用する項目を指定
    df_feature = df[feature_list]

    # ラベル特徴量をワンホットエンコーディング
    df_feature = pd.get_dummies(df_feature)

    # データセットを trainとtestに分割
    df_train = df_feature[df_feature['Survived'].notnull()]
    df_test = df_feature[df_feature['Survived'].isnull()].drop('Survived',axis=1)

    train_x = df_train.drop("Survived", axis=1)
    train_y = df_train[["Survived"]] # 2重[]

    test_x = df_test

    return train_x, train_y, test_x

    # # 使用する特徴量
    # df = df_base[feature_list]

    # # ラベル特徴量をワンホットエンコーディング
    # x = pd.get_dummies(df)

    # print(x)

    # #x = pd.concat([df_one, df_base[["Age", "Pclass", "Fare", "Title"]]], axis=1)
    # id = df_base[["PassengerId"]]
    # if is_train:
    #     y = df_base[["Survived"]]
    #     return x, y, id
    # else:
    #     return x, id

# ------------ Age ------------
from sklearn.ensemble import RandomForestRegressor
# Age を Pclass, Sex, Parch, SibSp からランダムフォレストで推定
def calc_feature_of_age(df, feature_list):

    # 推定に使用する項目を指定
    age_df = df[['Age', 'Pclass','Sex','Parch','SibSp']]

    # ラベル特徴量をワンホットエンコーディング
    age_df = pd.get_dummies(age_df)

    # 学習データとテストデータに分離し、numpyに変換
    known_age = age_df[age_df.Age.notnull()].values  
    unknown_age = age_df[age_df.Age.isnull()].values

    # 学習データをX, yに分離
    X = known_age[:, 1:]  
    y = known_age[:, 0]

    # ランダムフォレストで推定モデルを構築
    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
    rfr.fit(X, y)

    # 推定モデルを使って、テストデータのAgeを予測し、補完
    predictedAges = rfr.predict(unknown_age[:, 1::])
    #print(df_test.Age.isnull())
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges 
    #df_test = df_all[df_all['Survived'].isnull()].drop('Survived',axis=1)

    feature_list.append("Age")

    return df, feature_list

# 名前の特徴量作成
def calc_feature_of_name(df, feature_list):
    # Nameから敬称(Title)を抽出し、グルーピング
    df['Title'] = df['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
    df['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
    df['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
    df['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
    df['Title'].replace(['Mlle'], 'Miss', inplace=True)
    df['Title'].replace(['Jonkheer'], 'Master', inplace=True)
    #sns.barplot(x='Title', y='Survived', data=df, palette='Set3')

    # ------------ Surname ------------
    # NameからSurname(苗字)を抽出
    df['Surname'] = df['Name'].map(lambda name:name.split(',')[0].strip())

    # 同じSurname(苗字)の出現頻度をカウント(出現回数が2以上なら家族)
    df['FamilyGroup'] = df['Surname'].map(df['Surname'].value_counts()) 

    # 家族で16才以下または女性の生存率
    Female_Child_Group = df.loc[(df['FamilyGroup']>=2) & ((df['Age']<=16) | (df['Sex']=='female'))]

    Female_Child_Group = Female_Child_Group.groupby('Surname')['Survived'].mean()
    #print(Female_Child_Group.value_counts())

    # 家族で16才超えかつ男性の生存率
    Male_Adult_Group = df.loc[(df['FamilyGroup']>=2) & (df['Age']>16) & (df['Sex']=='male')]
    Male_Adult_List = Male_Adult_Group.groupby('Surname')['Survived'].mean()
    #print(Male_Adult_List.value_counts())

    # デッドリストとサバイブリストの作成
    Dead_list = set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
    Survived_list = set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

    # デッドリストとサバイブリストの表示
    # print('Dead_list = ', Dead_list)
    # print('Survived_list = ', Survived_list)

    # デッドリストとサバイブリストをSex, Age, Title に反映させる
    # 生死の典型パターンに書き換える
    df.loc[(df['Survived'].isnull()) & (df['Surname'].apply(lambda x:x in Dead_list)),\
                ['Sex','Age','Title']] = ['male',28.0,'Mr']
    df.loc[(df['Survived'].isnull()) & (df['Surname'].apply(lambda x:x in Survived_list)),\
                ['Sex','Age','Title']] = ['female',5.0,'Mrs']
    
    feature_list.extend(["Sex", "Title"])

    return df, feature_list

# 欠損値を Embarked='S', Pclass=3 の平均値で補完
def calc_feature_of_fare(df, feature_list):
    fare = df.loc[(df['Embarked'] == 'S') & (df['Pclass'] == 3), 'Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare)

    feature_list.append("Fare")
    return df, feature_list

# Family = SibSp + Parch + 1 を特徴量とし、グルーピング
def calc_feature_of_SibSp_Parch(df, feature_list):
    # 2～4：生存大
    # 1, 5,6,7：中
    # 8以上:小
    df['Family'] = df['SibSp'] + df['Parch'] + 1
    df.loc[(df['Family']>=2) & (df['Family']<=4), 'Family_label'] = 2
    df.loc[(df['Family']>=5) & (df['Family']<=7) | (df['Family']==1), 'Family_label'] = 1  # == に注意
    df.loc[(df['Family']>=8), 'Family_label'] = 0

    feature_list.append("Family_label")
    return df, feature_list

# ----------- Ticket ----------------
# 生存率で3つにグルーピング
def calc_feature_of_ticket(df, feature_list):
    # 同一Ticketナンバーの人が何人いるかを特徴量として抽出
    Ticket_Count = dict(df['Ticket'].value_counts())
    df['TicketGroup'] = df['Ticket'].map(Ticket_Count)
    #sns.barplot(x='TicketGroup', y='Survived', data=df, palette='Set3')
    #plt.show()

    # 生存率で3つにグルーピング
    df.loc[(df['TicketGroup']>=2) & (df['TicketGroup']<=4), 'Ticket_label'] = 2
    df.loc[(df['TicketGroup']>=5) & (df['TicketGroup']<=8) | (df['TicketGroup']==1), 'Ticket_label'] = 1  
    df.loc[(df['TicketGroup']>=11), 'Ticket_label'] = 0
    #sns.barplot(x='Ticket_label', y='Survived', data=df, palette='Set3')
    #plt.show()

    feature_list.append("Ticket_label")

    return df, feature_list

# Cabinの先頭文字を特徴量とする(欠損値は U )
def calc_feature_of_cabin(df, feature_list):
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Cabin_label'] = df['Cabin'].str.get(0)
    #sns.barplot(x='Cabin_label', y='Survived', data=df, palette='Set3')
    #plt.show()
    feature_list.append("Cabin_label")

    return df, feature_list

# 欠損値は一番乗船者が多かったSで補完
def calc_feature_of_embarked(df, feature_list):
    df['Embarked'] = df['Embarked'].fillna('S') 
    feature_list.append("Embarked")

    return df, feature_list

DEUBG = 1

if __name__ == '__main__':

    # データ取得
    df_train = pd.read_csv("../../data/input/train.csv") # 学習データ
    df_test = pd.read_csv("../../data/input/test.csv")   # テストデータ

    # テストデータを学習データの項目にそろえる
    df_test['Survived'] = np.nan

    # 学習とテストを混合
    df = pd.concat([df_train, df_test], ignore_index=True, sort=False)

    # 特徴量探索
    feature_list = ["Survived", "Sex", "Pclass"]

    # testデータの欠損Ageをランダムフォレストで補間
    df, feature_list = calc_feature_of_age(df, feature_list)

    # 名前から生存率を出して、そのグループのAge, Sex, Titleを置き換え
    df, feature_list = calc_feature_of_name(df, feature_list)

    # 欠損値を Embarked='S', Pclass=3 の平均値で補完
    df, feature_list = calc_feature_of_fare(df, feature_list)

    # 家族人数でグループ化
    df, feature_list = calc_feature_of_SibSp_Parch(df, feature_list)

    # Ticket番号でグループ化
    df, feature_list = calc_feature_of_ticket(df, feature_list)

    # Cabin
    df, feature_list = calc_feature_of_cabin(df, feature_list)

    # Embarked
    df, feature_list = calc_feature_of_embarked(df, feature_list)

    feature_list = list(set(feature_list))
    print(feature_list)

    # データセット作成
    x_train, y_train, x_test = create_dataset(df, feature_list)

    # 最適化パラメータ探索
    if DEUBG == 0:
        best_trial = util.optimize(util.objective_func, x_train, y_train)
        print(best_trial.params)
        print(util.params)

        util.params.update(best_trial.params)
    else:
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
    imp, metrics, model_list = util.train_cv(x_train, y_train, util.params, n_splits=5)

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
    my_solution.to_csv("../../data/output/003/submission.csv", index_label = ["PassengerId"])