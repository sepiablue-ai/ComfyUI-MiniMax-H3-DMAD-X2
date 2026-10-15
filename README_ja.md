# ComfyUI MiniMax H3 DMAD X2

日本語 | [English](README.md)

**公式のFL2VAモデルとDMAD LoRAから、自分のPCで4step融合モデルを作って使うComfyUIパッケージです。RTX 4070 12GBで、1088×1920・音声付きの5秒動画を約1分で生成します。** 配布済みの融合モデルに頼らない、もう1つの4step構成です。

| FL2VA・544×960生成 → 1088×1920出力・124フレーム | Case 01 | Case 02 |
|---|---:|---:|
| MATLOW融合版（[ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream)の例） | 62.6秒 | 58.7秒 |
| **本repo：DMAD融合版** | **60.5秒** | **57.0秒** |

[さやかベンチマーク](https://sayakabenchmark.pages.dev/)のFL2VAタブの実測です（各1回）。MATLOW融合版はFL2VA [022](https://sayakabenchmark.pages.dev/benchmarks/fl2va/022-matlow-fused4-sla5-x2vae-int8-stream/workflow.json)（2026-10-02）、本repoはFL2VA [014](https://sayakabenchmark.pages.dev/benchmarks/fl2va/014-dmad-fused4-sla5-x2vae-int8-stream/workflow.json)（2026-10-07）で、計測日が異なります。ComfyUI 0.38.0での別の計測セット（各3回の中央値）でも60.7秒と61.2秒で、**速度はほぼ同じ**です（[検証と生成時間](#検証と生成時間)）。

**用語**

| 用語 | 意味 |
|---|---|
| DMAD | 音声付き動画モデルを少ないstepで生成できるようにする蒸留手法。本repoは[Kijai変換](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/loras)の平均rank39 LoRAを使用 |
| 融合（fuse） | LoRAをモデルの重みに焼き込み、INT8に戻して1つのモデルファイルにすること。setupがGPUで一度だけ行う |
| SLA keep 5% | ComfyUIネイティブのスパースattention。重要な部分だけ（約5%）を計算する |
| X2 VAE INT8 streaming | 544×960のlatentを1088×1920に復号しながら、NVENCで動画保存を並行する |

> 作者：かみもと（Kamimoto）｜検証結果の速報は **[X（@sep_is_heim）](https://x.com/sep_is_heim)**、解説記事は **[note](https://note.com/sepiablue)**（[記事INDEX](https://note.com/sepiablue/n/n8e612c619a3e)）で公開しています。

ComfyUIで先頭画像から映像と音声を生成します。2つの例は544×960で生成し、1088×1920で保存します。124フレーム、24fpsです。

モデル融合ツール、導入処理、GUI/APIワークフローを同梱します。[ComfyUI-MiniMax-H3-PDMD-X2](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-PDMD-X2)と同じく、設定ファイルを用意して`setup.bat`から導入・モデル準備を行う構成です。repo自体は`ComfyUI/custom_nodes`の外に置いてください。追加カスタムノードの[ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream)は別途、選択したComfyUIへ導入します。

本パッケージの対象は**FL2VAベースだけ**です。融合モデル1本と、先頭画像1枚のI2VA例を2つ用意します。先頭・終端の両画像を使うFL2VAは未検証です。公式DMADの学習対象はT2VAなので、この画像条件付けは実験的な転用です。平均rank39変換、INT8融合、低解像度生成、X2、SLAの組み合わせについて、公式rank128の品質評価をそのまま適用しません。

## 同梱物

- `setup.bat` / `scripts/`：導入、入力素材の配置、GPU融合、既存モデルの検証・再利用。
- `config.example.json`：ComfyUI、モデル、出力先、共有モデルフォルダ、GPUの設定例。ベースはFL2VAに固定。
- `model_recipe.json`：対応モデルと実装のハッシュ、融合方式、生成条件。
- `workflows/` / `workflows/api/`：FL2VA融合モデルを使う先頭画像1枚I2VAのGUI/API計2組。
- `examples/assets/`：小さな入力画像。
- `docs/validation.md`：新規導入検証の結果、過去の計測条件、本パッケージの確認範囲。
- `LICENSE` / `NOTICE.md`：コードのライセンスと出典。

## 必要環境

- Windows、ComfyUI **0.37.0**、commit `1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7`。融合・サンプリングに使う実装のハッシュを検査し、相違時は停止します。既存coreは更新・変更しません。
- そのComfyUI専用venv、またはWindows PortableのPython。検証環境はPython 3.13.13、PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130、comfy-kitchen 0.2.35、comfy-aimdo 0.5.5、PyAV 18.1.0、safetensors 0.8.0、triton-windows 3.8.0.post28です。全依存の固定値は`constraints-windows.txt`にあります。
- NVIDIA CUDA GPUとH.264 NVENC。検証機はRTX 4070 12GB、システムRAM 48GiBです。48GiBは検証機の容量であり、必要最小値ではありません。
- Git。X2 Streamはcommit `eb148a3d0f91a273668bbe027bfcbafb85f2c5e8`を使用します。既存版に相違や編集があれば、上書きせず停止します。
- 新規融合には、元モデルとは別に**空き30GB**。setupが作る融合モデルはFL2VAの1本、約20.97GB／19.53GiBです。検証済みモデルの再利用なら、追加の融合モデルは作りません。元重み6ファイルの取得から始める完全新規環境は、元重み・2種の準備済みモデル・依存環境・キャッシュを含めて**空き約90GB**を目安にしてください。

**DMAD融合は設定したCUDA GPUで実行します。PDMD版のCPU融合とは異なります。** 1層ずつ処理し、過去のPyTorch割り当てピークは約1.30GiBでした。デバイス全体の使用量ではありません。融合中はそのGPUでの動画生成を止めてください。

新しい環境を用意する場合の例です。

```powershell
git clone https://github.com/Comfy-Org/ComfyUI.git D:\ComfyUI-DMAD
git -C D:\ComfyUI-DMAD checkout --detach 1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7
py -3.13 -m venv D:\ComfyUI-DMAD\.venv
& D:\ComfyUI-DMAD\.venv\Scripts\python.exe -m pip install --no-cache-dir torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

`py`にPython 3.13が登録されておらず、`uv`を導入済みなら、coreのclone・checkout後に次の方法を使えます。ベースPythonもComfyUIフォルダ内に置き、ユーザー共通の起動用ファイルやレジストリ登録を作りません。

```powershell
uv python install 3.13.13 --install-dir D:\ComfyUI-DMAD\python --no-bin --no-registry --no-cache
uv venv --seed --python D:\ComfyUI-DMAD\python\cpython-3.13.13-windows-x86_64-none\python.exe D:\ComfyUI-DMAD\.venv
uv pip install --python D:\ComfyUI-DMAD\.venv\Scripts\python.exe --no-cache torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

setupは`uv`があれば使い、なければ指定Pythonの`pip`を使います。既存torch/torchvision/torchaudioを固定し、記録した依存制約を適用します。グローバルPythonは拒否します。モデルのダウンロード、ComfyUIの起動、動画生成は行いません。

## モデル

モデルごとの利用条件に従って用意してください。重みは同梱していません。

| ファイル | models以下 | 用途・入手先 |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models/` | 未融合のFL2VAベース。[Comfy-Org H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| `minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors` | `loras/` | 対応する約367MBの[Kijai変換版](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/loras)。strength 1.0 |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders/` | [Comfy-Org H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `vae/` | 画像条件付け用の通常H3 VAE |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/` | 音声デコード。[Comfy-Org VAEs](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `MiniMax-H3-X2-Detail-v1.safetensors` | `vae/` | 変換元の[X2 Detail VAE](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE) |
| `MiniMax-H3-X2-Detail-v1-decoder-int8-convrot.safetensors` | `vae/h3_x2_stream/` | 準備済みX2 INT8 decoder、約2.83GB |

既存の変換済みX2 decoderは共有できます。未作成ならsetupでX2 Streamを導入した後、その`examples/00_prepare_int8.json`で一度変換してください。本パッケージの生成例は変換済みdecoderを直接読みます。**画像条件付けは通常H3 VAE、最終動画デコードはX2 VAE**を使います。

現在、公式にも[ComfyUI用ノードとrank128 LoRA](https://github.com/Yzmblog/DMAD/blob/main/comfyui/ComfyUI-DMAD/README.md)があります。本パッケージは既に検証した平均rank39版とnative LCMを使用します。融合ツールは指定以外のLoRAを拒否するため、公式rank128版をそのまま差し替えることはできません。

## DMADモデルを準備する

本パッケージを`ComfyUI/custom_nodes`の外へcloneし、そのフォルダから以下のsetupコマンドを実行します。

```powershell
git clone https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-DMAD-X2.git D:\H3-DMAD-X2
Set-Location D:\H3-DMAD-X2
```

1. `config.example.json`を`config.local.json`にコピーします。
2. `comfyui_root`、`base_model`、`dmad_lora`を設定し、`output_directory`と`gpu`を指定します。ベースは指定の未融合FL2VAファイルです。GPUは物理番号で、既定値は0です。相対パスは設定ファイルの場所を基準に解決します。`model_roots`には`vae`、`text_encoders`などを含む**modelsフォルダ**を追加できます。
3. 選択したComfyUIのPythonで実行します。

```bat
setup.bat "D:\ComfyUI-DMAD\.venv\Scripts\python.exe" "config.local.json"
setup.bat "D:\ComfyUI-DMAD\.venv\Scripts\python.exe" "config.local.json" --check
```

通常setupは依存を導入し、既存融合モデルがあれば融合記録・メタデータ・全ファイルSHA-256を検査して再利用します。なければ`minimax_h3_fl2va_DMAD_full_avg39_bf16merge_int8_convrot.safetensors`を作ります。200個のINT8 ConvRot重みと8個のBF16重みにLoRAを適用し、nativeのモジュール名由来seedで再量子化します。非変更テンソルの保持と代表6層の再読込も確認します。元モデルは保存したままです。

融合記録の既定位置はモデル横の`<モデル名>.fusion.json`です。**既存のFL2VA版DMADモデルを再利用する場合**は、`output_directory`を現在の保存先にし、`fusion_manifest`に元の`merge_result.json`を指定してください。以前のFL2VA融合記録を読み込めるため、約21GBのモデルをコピー・再作成する必要はありません。モデル名は記載したFL2VA出力名を使います。記録の欠損・不一致、無関係な出力、未完了ファイルがあれば停止します。

設定例にモデル種別の選択項目はありません。以前の設定に`model_kind: fl2va`があっても使えますが、ほかの種別は導入・融合前に拒否します。ベースファイルの間違いもサイズ・ハッシュで拒否します。

setupは設定ファイルの横に`extra_model_paths.local.yaml`を作り、入力画像を`dmad_`付きで`ComfyUI/input/`直下に配置します。既存の異なる設定・画像は上書きしません。生成用ファイルが不足していれば一覧を表示します。準備後に`--check`を実行してください。`setup --check`は読取専用で、依存・融合モデル・入力・パス設定の準備完了を確認します。

融合ツールを直接実行する場合です。

```powershell
& 'D:\ComfyUI-DMAD\.venv\Scripts\python.exe' -B -X utf8 scripts/fuse_dmad.py --config config.local.json
& 'D:\ComfyUI-DMAD\.venv\Scripts\python.exe' -B -X utf8 scripts/fuse_dmad.py --config config.local.json --inspect
& 'D:\ComfyUI-DMAD\.venv\Scripts\python.exe' -B -X utf8 scripts/fuse_dmad.py --config config.local.json --verify
```

`--inspect`は元ファイルと実装を検査し、GPU実行・書込をしません。`--verify`は既存融合モデルと記録も検査します。**融合ツールの`--check`**はsetupの読取専用チェックと異なり、CUDAで代表6層を検査して`<融合記録名>.check.json`を作ります。完全な融合モデルは作らず、既存の記録は上書きしません。

## ComfyUIで生成する

普段の起動方法に`--extra-model-paths-config`と生成されたYAMLを指定するか、次の前景起動補助を使います。

```powershell
& 'D:\ComfyUI-DMAD\.venv\Scripts\python.exe' -B -X utf8 scripts/launch.py --config config.local.json --port 8198
```

設定GPUをこのプロセスだけに指定し、localhostと`--disable-pinned-memory`で起動します。使用中ポートは拒否し、Ctrl+Cで終了します。

| GUIワークフロー | 用途 |
|---|---|
| `workflows/fl2va_case01_sla5.json` | 教室の会話。FL2VA融合モデルを使う先頭画像1枚I2VA、seed43 |
| `workflows/fl2va_case02_sla5.json` | 剣の場面。同じ融合モデルを使う先頭画像1枚I2VA、seed43 |

GUIにドラッグしてモデル選択を確認してください。API版は`workflows/api/`です。画像を手動配置する場合はファイル名に`dmad_`を付け、`ComfyUI/input/`直下に置きます。出力はComfyUI出力先の`DMAD/`以下です。

設定は**LCM、記録した4step manual sigmas、video/audio shift 12/2、BasicGuider、SLA keep5%、FFN chunks8**です。native LCMはdenoisedを次sigmaでre-noiseし、H3は音声の別scheduleを扱います。DMADの更新規則に対応しますが、公式と同じ乱数生成順序やrank128の出力再現を保証しません。**融合済みDMADへのLoRA二重適用や、MATLOWなど別の蒸留融合モデルへの融合は避けてください。**

## 検証と生成時間

2026-10-06～07に、現行`setup.bat`を新規ComfyUI・専用venvで検証しました。元重み6ファイルを再ダウンロードし、FL2VA融合モデルとX2 decoderを新規作成して、同梱API例2本を変更なしで生成できています。全映像・音声デコードを通過し、全体時間は**64.078秒（1分04.078秒）／58.175秒（0分58.175秒）**でした。連続した機能検証であり、コールド反復ベンチマークではありません。[検証記録](docs/validation.md)と[今回の機械可読記録](docs/fresh-install.json)に詳細があります。

最新の教室シーン全体時間中央値は、現行MATLOW＋SLA5%が**60.667秒（1分00.667秒）**、DMAD＋SLA5%が**61.210秒（1分01.210秒）**です。各3回、毎回モデル・実行キャッシュを解放し、OS・コンパイルキャッシュは保持しています。サーバー起動は時間に含みません。ComfyUI0.38.0上の別の反復セットであり、この小差から安定した速度優位を判断しません。両方の例に対するコールド計測値でもありません。映像・音声の品質は人による確認が必要です。

## 問題が起きた場合

- ハッシュ・core実装が違う：指定の未融合ベース、平均rank39 LoRA、検証済みcoreを選んでください。自動差替えやcore変更はしません。
- 既存融合モデルの記録がない：元の融合記録を探し、`fusion_manifest`へ指定してください。名前を合わせるためだけの大きな再作成は不要です。
- X2 decoderがない：X2 Streamの準備ワークフローで一度変換するか、既存のmodelsフォルダを`model_roots`へ追加します。
- セレクタにモデルが出ない：生成したYAMLを起動時に読み込み、パス変更後はComfyUIを再起動します。
- SSD不足：新規融合1本は空き30GBを確認します。検証済みモデルの再利用なら追加作成しません。
- 入力・設定の競合や中断：表示されたファイルを確認してから再試行します。異なる既存ファイルや記録は上書きしません。
- 音声・映像品質：生成物を視聴してください。デコード検査や公式の評価値だけでは、この組み合わせの品質は判断できません。

## 作者・関連リポジトリ

かみもと（Kamimoto）です。RTX 4070 12GBで、MiniMax H3とComfyUIの高速化・省VRAM化を実測しながら公開しています。

- X：[@sep_is_heim](https://x.com/sep_is_heim) — step数・秒数・短縮率などの検証速報
- note：[かみもと｜ローカルAI生成](https://note.com/sepiablue) — 導入方法と実測の解説記事（[記事INDEX](https://note.com/sepiablue/n/n8e612c619a3e)）
- [さやかベンチマーク](https://sayakabenchmark.pages.dev/) — H3ワークフローの速度・画質比較

| リポジトリ | 内容 |
|---|---|
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | X2 VAEのINT8化と非同期NVENC保存。1088×1920の音声付き5秒動画を約60秒で生成 |
| [ComfyUI-MiniMax-H3-PDMD-X2](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-PDMD-X2) | PDMD 2step融合モデル。1088×1920の5秒動画を約44〜50秒 |
| [ComfyUI-H3-LBH-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-LBH-X2-Stream) | 低解像度3step＋高解像度1step。14秒動画で約25%短縮 |
| [ComfyUI-H3-AudioFirst-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-AudioFirst-X2-Stream) | 音声を先に作って固定し、金属的な響きを抑える2パス生成 |

## ライセンス

コードとワークフローの改変部分はGPL-3.0です。[LICENSE](LICENSE)、[NOTICE.md](NOTICE.md)を参照してください。モデル重みはそれぞれの利用条件に従います。
