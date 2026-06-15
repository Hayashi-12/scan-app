# scan-app

エネルギーデバイス操作技術評価システム  
模擬臓器の撮影ガイド + 画像自動解析

---

## 概要

iPhoneで模擬臓器を撮影し、PCで自動解析するシステムです。  
撮影画像はSupabase（クラウド）経由でPCに転送されます。

```
iPhone（撮影ガイド） → Supabase（クラウド） → Windows PC（自動解析）
```

---

## ファイル構成

```
scan_cloud/
├── index.html          # iPhoneで開く撮影ガイドWebアプリ
├── config.json         # Supabaseの接続情報（各自設定）
├── sync_download.py    # PCで起動する自動同期スクリプト
├── analyze_burn.py     # 焦げ・残存腫瘍の画像解析
├── scan_validator.py   # 撮影条件チェックモジュール
└── requirements.txt    # 必要なPythonライブラリ
```

---

## セットアップ

### 1. Pythonライブラリのインストール

```bash
pip install opencv-python numpy pandas
```

### 2. config.json を編集

```json
{
    "supabase_url": "https://xxxxx.supabase.co",
    "supabase_anon_key": "sb_publishable_...",
    "bucket_name": "scans",
    "poll_interval_sec": 5,
    "download_dir": "received_images",
    "output_dir": "output"
}
```

### 3. PCで同期スクリプトを起動

```bash
python sync_download.py
```

### 4. iPhoneでWebアプリを開く

```
https://Hayashi-12.github.io/scan-app/
```

---

## 使い方

1. PCで `sync_download.py` を起動したままにする
2. iPhoneのSafariでURLを開く
3. 被験者IDを入力（例: S001）→ カメラを起動
4. ガイド枠に模擬臓器を合わせる
5. 全項目が緑になったら撮影ボタンを押す
6. 自動でPCにダウンロード・解析される
7. `output/burn_metrics.csv` に結果が追記される

---

## 解析指標

| 指標 | 内容 |
|------|------|
| burn_ratio_to_organ | 臓器面積に対する焦げ面積の割合 |
| mild / moderate / severe | 焦げの3段階分類（軽度・中等度・重度）|
| burn_gradient_corr | 焦げのグラジュエーション（中心-周辺の濃さ相関）|
| residual_tumor_ratio | 臓器面積に対する残存腫瘍の割合 |
| burn_tumor_centroid_dist | 焦げと残存腫瘍の重心間距離 |

---

## 研究概要

**エネルギーデバイス操作に関する技術評価方法の確立**  
医工連携による「上手さ」の定量化と教育標準化のための評価基盤構築

- 指導教員: 植村宗則 准教授
- 担当: 林優成
- 所属: 熊本大学大学院
