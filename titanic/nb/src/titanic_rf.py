
import numpy as np
import pandas as pd
import seaborn as sns # Samuel Norman Seabornからとっている

# 自作ライブラリインポート
import util
import importlib
importlib.reload(util) # ライブラリ更新時に対応

from sklearn.pipeline import Pipeline

# バリデーション
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
# 評価指標
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import itertools
import numpy as np
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# データセット作成
def create_dataset(df, feature_list):
    # 推定に使用する項目を指定
    df_feature = df[feature_list]

    # ラベル特徴量をワンホットエンコーディング
    df_feature = pd.get_dummies(df_feature)
    #df_feature = df_feature.loc[:, ['Survived', "Title_Mr","Sex_female", "Pclass","Fare","Age", "Ticket_label","Cabin_label_U","Sex_male","Embarked_S","Embarked_C","Family_label"]]
    #print(feature_list)

    # データセットを trainとtestに分割
    df_train = df_feature[df_feature['Survived'].notnull()]
    df_test = df_feature[df_feature['Survived'].isnull()].drop('Survived',axis=1)

    # train_x = df_train.drop("Survived", axis=1)
    # train_y = df_train[["Survived"]] # 2重[]

    #print(df_train.head())

    # データフレームをnumpyに変換
    X = df_train.drop('Survived',axis=1).values
    y = df_train['Survived'].values
    y = y.astype('int')
    test_x = df_test.values    

    return X, y, test_x

# ------------ Age ------------
from sklearn.ensemble import RandomForestRegressor
# Age を Pclass, Sex, Parch, SibSp からランダムフォレストで推定
def calc_feature_of_age(df, feature_list):

    # 推定に使用する項目を指定
    # age_df = df[['Age', 'Pclass','Sex','Parch','SibSp']]
    age_df = df[['Age', 'Pclass','Sex','Title']]

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
def calc_feature_of_title(df, feature_list):
    # Nameから敬称(Title)を抽出し、グルーピング
    df['Title'] = df['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
    df['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
    df['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
    df['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
    df['Title'].replace(['Mlle'], 'Miss', inplace=True)
    df['Title'].replace(['Jonkheer'], 'Master', inplace=True)
    #sns.barplot(x='Title', y='Survived', data=df, palette='Set3')
    feature_list.extend(["Title"])
    return df, feature_list

def calc_feature_of_surname(df, feature_list):
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

# 名前の特徴量作成
def calc_feature_of_surname2(all_data, feature_list):
    all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())
    Surname_Count = dict(all_data['Surname'].value_counts())
    all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])
    Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]
    Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]

    Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
    Female_Child.columns=['GroupCount']

    Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
    Male_Adult.columns=['GroupCount']

    Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
    Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
    print(Dead_List)
    Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
    Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

    train=all_data.loc[all_data['Survived'].notnull()]
    test=all_data.loc[all_data['Survived'].isnull()]
    test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
    test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
    test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
    test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
    test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
    test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'    

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

# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'

#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     fig, ax = plt.subplots(figsize=(10, 10))
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, fontsize=25)
#     plt.yticks(tick_marks, fontsize=25)
#     plt.xlabel('Predicted label',fontsize=25)
#     plt.ylabel('True label', fontsize=25)
#     plt.title(title, fontsize=30)
    
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size="5%", pad=0.15)
#     cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
#     cbar.ax.tick_params(labelsize=20)
    
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
# #            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), ha="right",
#              rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     fontsize=20,
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax


DEUBG = 1

def train_rf_cv(input_x, input_y, n_splits):
    # 結果格納用
    metrics = []
    imp = pd.DataFrame()
    model_list = []

    # K分割検証法で学習用と検証用に分ける
    cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(input_x, input_y))

    # from sklearn.model_selection import GridSearchCV

    # param_grid = {"max_depth": [2, 3, 4, 5, None],
    #               "n_estimators":[1, 3, 10, 30, 100],
    #               "max_features":["auto", None]}

    # model_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=10),
    #                 param_grid = param_grid,   
    #                 scoring="accuracy",  # metrics
    #                 cv = 10,              # cross-validation
    #                 n_jobs = 1)          # number of core

    # model_grid.fit(input_x, input_y) #fit

    # model_grid_best = model_grid.best_estimator_ # best estimator
    # print("Best Model Parameter: ", model_grid.best_params_)
    # Best Model Parameter:  {'max_depth': 5, 'max_features': None, 'n_estimators': 100}

    # モデル作成
    # 採用する特徴量を絞り込む
    select = SelectKBest(k = 10)

    # モデルパラメータ
    clf = RandomForestClassifier(random_state = 10, 
                                warm_start = True,  # 既にフィットしたモデルに学習を追加 
                                n_estimators = 26,
                                max_depth = 6, 
                                max_features = 'sqrt')
    
    # モデル作成
    pipeline = make_pipeline(select, clf)
    
    # ループ回数分 RFを試す
    for nfold in np.arange(n_splits):
        # 区切り線
        print("-"*20, nfold, "-"*20)

        # 学習データ、検証データのインデックスを取得
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        
        # インデックスのデータを取得
        x_tr, y_tr = input_x[idx_tr.tolist()], input_y[idx_tr.tolist()]
        x_va, y_va = input_x[idx_va.tolist()], input_y[idx_va.tolist()]
        # print("x_train", x_tr.shape, "y_valid", y_tr.shape)
        # print("x_valid", x_va.shape, "y_valid", y_va.shape)

        # 学習
        pipeline.fit(x_tr, y_tr)
        model_list.append(pipeline)

        # 推定
        y_tr_pred = pipeline.predict(x_tr)
        y_va_pred = pipeline.predict(x_va)

        # Yデータの偏り確認
        print("y_train:{:.3f}, y_tr:{:.3f}, y_va:{:.3f}".format(
            input_y.mean(),
            y_tr.mean(),
            y_va.mean(),
        ))

        # 正解と予測から正解率を算出
        metric_tr = accuracy_score(y_tr, y_tr_pred)
        metric_va = accuracy_score(y_va, y_va_pred)
        print("[accuracy] tr: {:.2f}, va: {:.2f}".format(metric_tr, metric_va))   

        # 全体結果に格納
        metrics.append([nfold, metric_tr, metric_va])
        
    # まとめ結果を表示
    print("-"*20, "result", "-"*20)
    metrics = np.array(metrics)
    print(metrics)

    # 正確性の平均、偏差
    print("[cv ] tr: {:.2f}+-{:.2f}, va: {:.2f}+-{:.2f}".format(
        metrics[:,1].mean(), metrics[:,1].std(),
        metrics[:,2].mean(), metrics[:,2].std(),
    ))

    print("Done.")
    
    return metrics, model_list

def classifier_accuracy(model, X_test, y_test):
    
    #the CV 5 fold score
    x=cross_val_score(model, X_test,y_test,cv=5).mean()*100
    print(f"5 Fold CV Accuracy = {x:.2f}%\n\n")
    
    #The ROC curve
    print("The ROC: ")
    y_probs=model.predict_proba(X_test)
    y_probs_positive=y_probs[:,1]
    fpr,tpr,thresholds=roc_curve(y_test,y_probs_positive)
    plt.plot(fpr,tpr,color="orange",label='ROC')
    plt.plot([0,1],[0,1],label="No predictive Power Line",linestyle='--')
    print(f"AUC: {roc_auc_score(y_test,y_probs_positive)*100:.2f}%")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    plt.show()
    
    #The confusion matrix
    print("The Confusion Matrix: ")
    y_preds=model.predict(X_test)
    conf_mat=confusion_matrix(y_test,y_preds)
    sns.heatmap(data = conf_mat,annot=True,fmt='d')
    plt.show()
    
    #classidication report
    print("Classification Report:-\n")
    print(classification_report(y_test,y_preds))

def train_rf_my_cv(X, y, test_x):
    # CV実行
    n_splits = 10
    metrics, model_list = train_rf_cv(X, y, n_splits)

    # 結果を辞書に保存
    solution = {}
    
    # 各モデルで予測
    for i, model in enumerate(model_list):
        solution[str(i) + "_model"] = model.predict(test_x)

    # 辞書からDataFrameに変更
    solution = pd.DataFrame(solution)

    # 多数決 (最頻値)を取得
    solution_max = solution.mode(axis = 1).values
    # なぜか Nanが2列目についてくるため2列目を削除
    solution_max = [[int(x[0])] for x in list(solution_max)]

    predictions = [x[0] for x in solution_max]

    return predictions

def train_rf_lib_cv(X, y, test_x):

    # パラメータチューニング
    # pipe = Pipeline([('select',SelectKBest(k=20)), 
    #             ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])    
    
    # param_test = {'classify__n_estimators':list(range(20,50,2)), 
    #           'classify__max_depth':list(range(3,10,3))}

    # gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)
    # gsearch.fit(X,y)
    # print(gsearch.best_params_, gsearch.best_score_)


    # 採用する特徴量を25個から20個に絞り込む
    select = SelectKBest(k = 20)

    clf = RandomForestClassifier(random_state = 10, 
                                warm_start = True,  # 既にフィットしたモデルに学習を追加 
                                n_estimators = 26,
                                max_depth = 9, 
                                max_features = 'sqrt')
    
    pipeline = make_pipeline(select, clf)
    pipeline.fit(X, y)

    classifier_accuracy(pipeline, X, y)

    # フィット結果の表示
    cv_score = cross_val_score(pipeline, X, y, cv= 10)
    print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

    #cv_result = cross_validate(pipeline, X, y, cv = 10)
    #print('mean_score = ', np.mean(cv_result['test_score']))
    #print('mean_std = ', np.std(cv_result['test_score']))

    #conf_mat = confusion_matrix(y, cv_result)
    #print(conf_mat)
    #plot_confusion_matrix(conf_mat, list(iris.target_names))

    # --------　採用した特徴量 ---------------
    # 採用の可否状況
    mask = select.get_support()

    # 項目のリスト
    list_col = list(df.columns[1:])

    # 項目別の採用可否の一覧表
    for i, j in enumerate(list_col):
        print('No'+str(i+1), j,'=',  mask[i])

    # シェイプの確認
    X_selected = select.transform(X)
    print('X.shape={}, X_selected.shape={}'.format(X.shape, X_selected.shape))

    # 推測
    predictions = pipeline.predict(test_x)

    return predictions

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

    # 敬称をまとめる
    df, feature_list = calc_feature_of_title(df, feature_list)

    # testデータの欠損Ageをランダムフォレストで補間
    df, feature_list = calc_feature_of_age(df, feature_list)
    
    # 名前から生存率を出して、そのグループのAge, Sex, Titleを置き換え
    df, feature_list = calc_feature_of_surname(df, feature_list)
    #df, feature_list = calc_feature_of_surname2(df, feature_list)
    

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
    X, y, test_x = create_dataset(df, feature_list)

    # print(X)
    # print(y)
    # print(test_x)

    # ----------- 推定モデル構築 ---------------
    file_name = ""
    if 0:
        predictions = train_rf_my_cv(X, y, test_x)
        file_name = "../../data/output/003/submission_rf_cv.csv"
    else:
        predictions = train_rf_lib_cv(X, y, test_x)
        file_name = "../../data/output/003/submission_rf.csv"

    # ----- Submit dataの作成　------- 
    PassengerId = df_test['PassengerId']
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions})
    submission.to_csv(file_name, index=False)

