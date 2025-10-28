# Anime-Llasa-3B-Captions-Demo ローカル版
このリポジトリは https://huggingface.co/spaces/OmniAICreator/Anime-Llasa-3B-Captions-Demo をフォークしたものです。

[Anime Llasa 3B Captions model](https://huggingface.co/NandemoGHS/Anime-Llasa-3B-Captions) を動かす Huggingface spaces 用のプログラムをローカルで動くよう修正しています。

元のライセンスがわからないので、このリポジトリの著作者の修正はパブリックドメイン扱いにします。

[自作のパッチ](https://gist.github.com/asfdrwe/c9fd1fe8aeb69fa90d5865d761f59eeb)と[5chのコード1](https://files.catbox.moe/wxfdul.py)と[5chのコード2](https://files.catbox.moe/6lm1wv.py)と[5chのコード3](https://files.catbox.moe/tj9z74.txt)を元に修正しています。

[元のREADME.md](README-original.md)

## Windows

あらかじめ [git](https://gitforwindows.org/) と [python3](https://www.python.org/downloads/windows/) をインストールしてください。python のバージョンは 3.12.x がおすすめです。

### インストール
ターミナルを起動し、git でリポジトリをダウンロードし、venv で仮想環境を作り、必要なモジュールを pip でインストールします。
```
git clone https://github.com/asfdrwe/Anime-Llasa-3B-Captions-Demo
cd Anime-Llasa-3B-Captions-Demo
python -m venv venv
.\venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

[Windows版ffmpeg バイナリ配布サイト](https://www.gyan.dev/ffmpeg/builds/)から[ffmpeg-7.1.1-full_build-shared.7z](https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-full_build-shared.7z)をダウンロードし、エクスプローラーで右クリックしてすべて展開してください。

展開されたフォルダを開き、binフォルダを開き、中にあるすべてのファイルを Anime-Llasa-3B-Captions-Demo のあるフォルダ内の `venv\Lib\site-packages\torchcodec` にコピーしてください。


### 実行

```
python app.py
```
もしくは`run.bat`をダブルクリックしてください。

VRAM 12GB の GPU では参照音声使用時に VRAM があふれる可能性があります。
--move-model オプションを付けると Whisper モデルを使用する前に Llasa モデルを一旦
VRAM から退避し、Whisper モデル実行後に戻すことで、VRAM あふれを防ぐことができます。
```
python app.py --model-move
```
もしくは`run-model-move.bat`をダブルクリックてください。

遅いですが完全にCPUだけで動かすことも可能です。`--full-cpu`オプションをつけて起動してください。
```
python app.py --full-cpu
```
もしくは`run-full-cpu.bat`をダブルクリックして起動してください。

自動的にブラウザが開きます。

## Linux
git と python3.12(または3.13) と ffmpeg をインストールしてください。


### インストール
git でインストールし、venv で仮想環境を作り、pip で必要なモジュールをインストールします。
```
git clone https://github.com/asfdrwe/Anime-Llasa-3B-Captions-Demo
cd Anime-Llasa-3B-Captions-Demo
python -m venv venv
. venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

上記はGeforceの場合です。Radeon rocm6.4 の場合はcu128の代わりにrocm6.4にしてください。
```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

### 実行

```
python app.py
```

VRAM 12GB の GPU では参照音声使用時に VRAM があふれる可能性があります。
--move-model オプションを付けると Whisper モデルを使用する前に Llasa モデルを一旦
VRAM から退避し、Whisper モデル実行後に戻すことで、VRAM あふれを防ぐことができます。
```
python app.py --model-move
```

遅いですが完全にCPUだけで動かすことも可能です。`--full-cpu`オプションをつけて起動してください。
```
python app.py --full-cpu
```

自動的にブラウザが開きます。

## 変更履歴
- 2025/10/28
  - [5chのコード3](https://files.catbox.moe/tj9z74.txt) を元に、参照音声使用時に Llasa モデルを一旦 VRAM から退避し Whisper モデルを実行して参照音声の内容を文章で取得したあと、Whisper モデルを VRAM から退避させて Llasa モデルを VRAM に戻す機能を取り込み。--model-move オプション使用時に有効化
  - 上記機能で代替できるので --whisper-cpu オプションを削除
- 2025/10/26
  - 文書を日本語化
  - 下記の Anime-XCodec2 を使ったワークアラウンドを削除し、44.1KHz で動くよう[修正したコード](https://files.catbox.moe/6lm1wv.py)を取り込み
  - VRAM 12GB だと参照音声使用時にVRAMがあふれるのでwhisperをCPUで動かす--whisper-cpu オプションを追加
  - 完全にCPUだけで動かす--full-cpu オプションを追加
- 2025/10/25
  - 参照音声使用時に NandemoGHS/Anime-XCodec2-44.1kHz ではおかしいので、 NandemoGHS/Anime-XCodec2 を使用して 16kHz で音声生成するよう変更
  - bfloat16 で moderl を動かすよう変更
  - ブラウザを自動起動するよう修正
  - windows 向けに run.bat を追加
