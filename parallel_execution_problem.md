まずコードベースを調査します。

Ran terminal command: find /home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence -type f -name "*.py" | sort

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/commands/step3_collect.sh)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/execution/manager.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/decoding/result_collection.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/execution/worker.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/execution/_execution_utils.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/execution/hashing.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/execution/pool_runner.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/decoding/__main__.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/execution/models.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/execution/batching.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/decoding/resume.py)

コードを十分に理解できました。問題の根本原因を分析して修正を実装します。

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/decoding/metadata.py)

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/decoding/incomplete.py)

全体像が把握できました。根本原因と修正方針が明確になりました。実装します。

**特定された問題:**
1. `collect_results()` が同一の `scan_parquet` lazy frameを複数回再評価するため、NFS上でのファイル再読み込みが複数回発生し、特に `.sort("shot_id").sink_parquet()` がブロック/ハングする
2. `convert_legacy_detail_stats()` が全バッチから同じ `output_dir` を同時スキャンし、同一ファイルに同時書き込みする競合状態
3. `metadata.json` / `incomplete_shots.json` が全バッチで共有パスに直接書き込まれる競合

Read [](file:///home/quantum_teresheys/workspace/decoder-confidence/src/decoder_confidence/decoding/result_collection.py)

問題が完全に把握できました。修正を実装します。次に `metadata.py` と `incomplete.py` の原子書き込み対応:変更を確認してからテストを実行します:

Ran terminal command: python -c "
from decoder_confidence.decoding.result_collection import collect_results, _atomic_parquet
from decoder_confidence.decoding.metadata import write_metadata
from decoder_confidence.decoding.incomplete import write_incomplete_shots
print('imports OK')
import inspect, polars as pl
src = inspect.getsource(collect_results)
assert 'read_parquet' in src and 'scan_parquet' not in src, 'collect_results must use read_parquet'
assert '_atomic_parquet' in src, 'collect_results must use _atomic_parquet'
print('collect_results: read_parquet + _atomic_parquet OK')
src2 = inspect.getsource(write_metadata)
assert 'os.replace' in src2, 'write_metadata must use os.replace'
print('write_metadata: atomic write OK')
src3 = inspect.getsource(write_incomplete_shots)
assert 'os.replace' in src3, 'write_incomplete_shots must use os.replace'
print('write_incomplete_shots: atomic write OK')
"


Ran terminal command: python -m pytest tests/ -x -q 2>&1 | head -60

全テスト通過。ではレポートを出力します。

---

## バグ調査・修正レポート

### 症状

複数の計算ノードがバッチ番号（1〜4）を別々に受け持って同時に実行した場合、稀に一部のバッチで：

- `decoder_stat_batch=N.parquet` が生成されない
- `chunks/batch=N/` 以下にチャンクファイルが残存したまま
- 計算ノード上のプロセスが実行し続けてスタックする

---

### 根本原因

#### 原因1（主要・ハング）: `sink_parquet` の遅延評価 × NFS競合

`collect_results()` は単一の `pl.scan_parquet(...)` lazy frame を作成し、それを以下のように**複数回再評価**していた：

```python
scan = pl.scan_parquet([chunk_paths...])
scan.select(...).sink_parquet("logicalerror_batch=N.parquet")   # NFS読み込み1回目
scan.select(...).sink_parquet("metric=forced_gap_ml_batch=N.parquet")  # 2回目
_write_detailed_stats(scan, ...)   # 3回目
_write_decoder_stats(scan, ...)    # 4回目 ← sort("shot_id") 付き
```

NFS上で複数ノードが同じ `output_dir` に同時書き込みを行うと、NFS サーバ側でディレクトリメタデータのロック競合が発生する。特に最後の `_write_decoder_stats` の `.sort("shot_id").sink_parquet()` は全データのソートを含む最もコストの高い操作であり、NFS の書き込み応答待ちでブロックした場合、プロセスが無限にハングする。

**チャンクが残る理由**: ハングは `collect_results()` 内で発生するため、`cleanup_intermediate()` が呼ばれず、チャンクが削除されない。

#### 原因2（競合・データ破損リスク）: `convert_legacy_detail_stats()` の並列実行

この関数は全バッチが完了時に **共有 `output_dir` 全体をスキャン**し、他バッチの legacy ファイルも処理しようとする。`target.exists()` のチェックは NFS の negative キャッシュにより信頼できず、複数ノードが同一ファイルを同時書き込みする競合が発生しうる。

#### 原因3（競合）: `metadata.json` / `incomplete_shots.json` の同時上書き

どちらも `output_dir/` 直下の固定パスに**直接書き込み**するため、複数バッチが同時に完了した際に書き込み競合が発生する。

---

### 修正内容

#### `result_collection.py`

| 変更点 | 修正前 | 修正後 |
|---|---|---|
| チャンク読み込み | `pl.scan_parquet(...)` lazy（N回再評価） | `pl.read_parquet(...)` eager（**1回のみ**読み込み） |
| parquet書き込み | `.sink_parquet(target_path)` 直接書き込み | `_atomic_parquet()` でtempファイル→`os.replace()` 原子書き込み |
| legacy変換 | legacy ファイルがなくても全バッチ分スキャン | **early return** + 原子書き込みで競合安全化 |
| `_write_detailed_stats` / `_write_decoder_stats` 引数 | `pl.LazyFrame` | `pl.DataFrame`（eager読み込み後に受け取る） |

**`_atomic_parquet()` の仕組み:**

```python
def _atomic_parquet(df, path):
    tmp = path.parent / f".tmp_{os.getpid()}_{path.name}"  # PID付きでユニーク
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)  # POSIX rename(2) = 原子操作
```

`os.replace()` は POSIX `rename(2)` にマップされ、NFS 上でも同一サーバ・ディレクトリ内では原子操作として保証される。これにより「書きかけのファイルが他ノードから見える」状態が解消される。

#### `metadata.py` / `incomplete.py`

`write_metadata()` および `write_incomplete_shots()` も同様の **tempファイル + `os.replace()`** パターンに変更。複数バッチが同時に完了した場合も、最後に rename したノードの内容が残る（どちらも valid）。

---

### 変更ファイル一覧

- [src/decoder_confidence/decoding/result_collection.py](src/decoder_confidence/decoding/result_collection.py) — 主要修正（eager read + 原子書き込み）
- [src/decoder_confidence/decoding/metadata.py](src/decoder_confidence/decoding/metadata.py) — 原子書き込み対応
- [src/decoder_confidence/decoding/incomplete.py](src/decoder_confidence/decoding/incomplete.py) — 原子書き込み対応

全83テスト通過済み。