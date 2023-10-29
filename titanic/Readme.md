# Titanic

# Log

## 2023.10.29

* Random Forest導入
    * Score: 0.78229

* 参考コピペ
    * Score: 0.80622
    * 安定している

* Todo
    * 参考通りにならない
    * スコアが安定しない。100%-80%
        * インデックスが直地だった[:, 1:]。Survivedの位置によってぶれていた。
    * 10回超えたので次回submitする
    

## 2023.10.28

* Fare
    * Fareを追加
    * Score: 0.79186

* Family
    * Score: 0.78708

* すべて追加
    * Score: 0.78947

* Todo
    * 前日継続
    * 特徴量を削る    

## 2023.10.27

* Name
    * 'Title'を特徴量に追加
    * Score: 0.76555。若干落ちる。

## 2023.10.26

* Name
    * 敬称や名字から特徴量としたが変わらず
    * Score: 0.77033

* Todo
    * 前日継続

## 2023.10.25

* コード整理

## 2023.10.23

* Age
    * test_dataのAgeをランダムフォレストで補完した
    * Score: 0.77033

* Todo
    * 前日継続

## 2023.10.22

* train, testのデータセット作成関数を実装

* Todo
    * 特徴量の変更
        * https://qiita.com/jun40vn/items/d8a1f71fae680589e05c を真似してみる



## 2023.10.21

utilモジュールをimportするところができた。
titanic.pyに必要部分を収めた

util.pyが更新された時の取り込みとしてreloadが必要だった。
>import importlib
>importlib.reload(util)

* Todo
    * 特徴量の変更
    * train, testのデータセットを同時に作る。検索であるはず


## 2023.10.20

共通関数を作成中
相対パスでうまくいかず、直下のnb/srcとした

D:\WorkSpace\Kaggle\titanic\nb\002_common.ipynb

## 2023.10.19

Kaggle notebookの内容をコピーした

D:\WorkSpace\Kaggle\titanic\nb\001_base.ipynb

## 2023.10.18

vscode環境に移植した

* pipとanacondaが競合している？
* anacondaのパッケージがnumpy 1.19.2で古い。
これを更新できない

* anacondaをアンインストールした



