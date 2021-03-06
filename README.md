# Histopathologic Cancer Detection
このレポジトリは[aaryapatel007/Histopathological-Cancer-Detection](https://github.com/aaryapatel007/Histopathological-Cancer-Detection)の改良版です。
[Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)に対するコードです。
教育用です。

## 準備
1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)をインストールしましょう。

2. 環境を準備しましょう。
```
conda create -n breast_cancer python=3.6
conda activate breast_cancer
```
3. レポジトリをクローンして、必要なライブラリをインストールしておきましょう。
```
git clone https://github.com/kztakemoto/breast_cancer.git
cd breast_cancer
pip install -r requirements.txt
```
4. [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)からデータをダウンロードします。
データは``histopathologic-cancer-detection``ディレクトリに格納されているとして、そのディレクトリを現在のディレクトリ（つまり``breast_cancer``）においておきます。

## データの整形
後で使いやすいように、画像データやラベルをnpyファイルに保存しておきます。``data``ディレクトリに格納されます。
```
python make_data.py
```

## モデルの訓練
[Octave-ResNet-50](https://docs.openvinotoolkit.org/latest/omz_models_model_octave_resnet_50_0_125.html)のモデルアーキテクチャを利用しています。
ImageNetで事前訓練された重みは利用していません。
モデル重みは``weight/octresnet_one_cycle_model.h5``で保存されます。
```
python train_model.py
```
検証データに対する予測精度（``val_acc``）は97%くらいになります。

## （参考）保存されたモデルを使って予測
保存されたモデル重みを読み込んで、予測を行うまでの一連になります。
```
python load_model_and_predict.py
```
