# relay-bp: 非収束時のconfidence指標・logical error判定の挙動

このドキュメントは、decoder に `relay-bp` を使う場合に、BP が収束しなかったケースを
各指標(`metric`)がどう扱うかを説明する。対象となる指標は次の4つ:

- `linearize_logicalgap` （`src/decoder_confidence/decoding/_linearize_logicalgap.py`,
  `LinearizeLogicalGapDecoder`）
- `forced_gap_ml` （`src/decoder_confidence/decoding/_forced_gap.py`,
  `ForcedGapMLDecoder`）
- `reweighted_linearized_gap`
  （`src/decoder_confidence/decoding/_reweighted_linearized_gap.py`,
  `ReweightedLinearizedGapDecoder`）
- `ar-pec` / `ar-lec`（`src/decoder_confidence/decoding/_argument_reweighting.py`,
  `ArgumentReweightingDecoder`）

これらはすべて decoder に依存しない共通実装で、`decoder_factory.py` がどの decoder が
指定されていても metric 名だけで振り分ける。非収束の概念があるのは `RelayBpDecoderAdapter`
（relay-bp）だけなので、各実装は `isinstance(self.adapter, RelayBpDecoderAdapter)` の
ときだけ以下の挙動を有効化する。それ以外の decoder（BP-LSD, MWPM, ILP, VIBE-LSD）では
このドキュメントの内容は一切適用されず、既存の動作から変化しない。

`forcing_degradation_test`（`_forcing_degradation_test.py`）は別実装・別指標であり、
本ドキュメントの対象外。

## 背景（経緯）

以前は非収束時の挙動を実装した専用モジュール `_relay_bp.py`
(`RelayBpMetricDecoder`, metric名 `linearized_logicalgap` など) が別に存在したが、
metric名の綴り違い（`linearized_logicalgap` の "d"）と `decoder_factory.py` の
ディスパッチ順序（metric名チェックが decoder名チェックより先に来る）により、
実際の config からは到達不能なデッドコードになっていた。例えば
`decoder: relay-bp` / `metric: linearize_logicalgap`
（"d" なし、すべての実 config が使っている綴り）は常にこの共通実装側へ流れ、
`_relay_bp.py` には一切届かない。

このため、`forced_unconverged_confidence_value` を設定しても
`Unsupported linearize_logicalgap metric option(s): forced_unconverged_confidence_value`
という `ValueError` になっていた（バグ）。`_relay_bp.py` で設計・実装・テスト済みだった
非収束時の挙動を、実際に到達する上記4ファイルへ移植し、`_relay_bp.py` 自体は削除した。

各実装は元々 `RelayBpDecoderAdapter` を使う際に `result.success` を無視して
`self.adapter.decode(...)`（生の decoding のみ返す）を呼んでいたため、非収束の shot も
収束した shot と全く同様に扱われ、フラグも立たなかった。これが実質的なバグ本体であり、
`ValueError` はその手前で config を弾いていただけだった。

## 2段階デコードの構成（おさらい）

- **stage1**: 通常の（制約なしの）decodeを1回行う。`converged1, c1, l1, w1` を得る
  (`converged1` は relay-bp の `decode_detailed_single(...).success`、`c1` は
  correction、`l1` は correction から求めた logical class、`w1` はその重み）。
- **stage2 (constrained, "2a")**: observable `i` ごとに、パリティ検査行列へ observable
  の行を追加し、syndrome に `1 - l1[i]` を追加した上で再度decodeする。収束した場合は
  必ず `l1` と observable `i` のビットが異なる correction が得られる（構成上保証）。
  `reweighted_linearized_gap` ではこれを "stage2a" と呼ぶ。
- **stage2b**（`reweighted_linearized_gap` のみ）: priorをbaselineのcorrectionに向けて
  再重み付けした上で、制約なしの decode をもう一度行う。stage2aと異なり、こちらは
  収束しても baseline と同じ logical class に落ち着く可能性がある。

## 非収束時の挙動まとめ

| ケース | linearize_logicalgap | forced_gap_ml | reweighted_linearized_gap | ar-pec / ar-lec |
|---|---|---|---|---|
| stage1 (baseline) が非収束 | `-np.inf` | `0.0` | `-np.inf` | `accept=False` |
| stage1 非収束時の logical error | **強制 True** | **強制 True** | **強制 True** | **強制 True** |
| stage1 非収束時の stage2 実行 | 行わない | 行わない | 行わない | （round0以降を実行しない） |
| stage2 が実質的に全滅 | config値 | config値 | config値 | 既存動作のまま（accept=Falseのみ） |
| stage2 全滅時の logical error | 上書きしない（stage1の予測で判定） | 同左 | 同左 | 同左 |
| stage2 に1つ以上、baselineと異なるlogical classの収束解がある | それを使って既存の式で計算 | 既存のML選択ロジックのまま | それを使って既存の式で計算（stage2a全滅でもstage2bが使える場合はそれを使う） | ラウンド間の一致判定（既存動作） |

「stage1非収束」と「stage2全滅+`negative`設定」はどちらも `-np.inf`（あるいは
`forced_gap_ml`なら`0.0`）になり、confidence値だけでは区別できない。これは意図した
仕様であり、区別用の補助カラムは追加していない。

## metric_options: `forced_unconverged_confidence_value`

`linearize_logicalgap` / `forced_gap_ml` / `reweighted_linearized_gap` の3指標では、
decoder が relay-bp の場合に限り `metric_options.forced_unconverged_confidence_value`
が **必須** になる（`LinearizeLogicalGapDecoder.__post_init__` /
`ForcedGapMLDecoder.__post_init__` / `ReweightedLinearizedGapDecoder.__post_init__` が
`isinstance(adapter, RelayBpDecoderAdapter)` かつ未指定なら `ValueError`）。値は
`"positive"` または `"negative"`（大文字小文字は無視、strip される）。他の decoder
（BP-LSD 等）では不要・無視される。

`ar-pec` / `ar-lec` はこのオプションを使わない（boolean の `accept`/`reject` のみで、
数値のconfidenceを持たないため）。

「stage2が実質的に全滅」した場合の値は次の通り:

| forced_unconverged_confidence_value | linearize_logicalgap | forced_gap_ml | reweighted_linearized_gap |
|---|---|---|---|
| `positive` | `np.inf` | `np.inf` | `np.inf` |
| `negative` | `-np.inf` | `0.0` | `-np.inf` |

yaml設定例:

```yaml
decoder: relay-bp
metric: linearize_logicalgap

decoder_options:
  set_max_iter: 30

metric_options:
  get_detail_stat: true
  forced_unconverged_confidence_value: negative
```

`get_detail_stat: true` は `linearize_logicalgap` / `forced_gap_ml` では
`forced_unconverged_confidence_value` と併用できる（下記「decoder_statsとの関係」参照）。

`reweighted_linearized_gap` の場合は既存必須オプション `b` と併記する:

```yaml
metric: reweighted_linearized_gap
metric_options:
  b: 2.0
  forced_unconverged_confidence_value: negative
```

## 各指標の詳細

### `linearize_logicalgap` / `forced_gap_ml`

```
stage1 非収束
  → pred = l1（非収束時のstage1予測をそのまま使う）, confidence = -inf(linearize) / 0(forced_gap_ml)
    logical error を強制 True、stage2 は実行しない

stage1 収束
  num_obs == 0、または stage2 の全observableが非収束
  → pred は変わらず、confidence = forced_unconverged_confidence_value に基づく値
    logical error への上書きはなし（stage1の予測 = l1 で判定される）

  stage2 に1つ以上収束したinstanceがある
  → 既存の計算式のまま（非収束のinstanceは候補から除外するだけ）
    - linearize_logicalgap: 収束した中の最小重み - w1
    - forced_gap_ml: [stage1解] + [収束したstage2解たち] をプールし、
      最小重みをML予測として採用（predがstage1と入れ替わることがある）
```

### `reweighted_linearized_gap`

stage2a（constrained、各observable強制反転）は収束すれば構成上必ず baseline と異なる
logical class になるが、stage2b（reweighted unconstrained）は収束しても baseline と
**同じ** logical class になり得る。そのため「stage2が実質的に全滅したか」の判定は
stage2a単独ではなく、次の条件で行う:

```python
have_2a = 収束したstage2aインスタンスが1つ以上ある
have_2b_diff = stage2bが収束 かつ baselineと異なるlogical class

if not have_2a and not have_2b_diff:
    confidence = forced_unconverged_confidence_value に基づく値
    logical errorへの上書きなし
```

`have_2a` が False でも `have_2b_diff` が True であれば、既存のgap計算式
（`gap = min(best_w_constrained, w_r) - w1` など）はそのまま利用できる。
`best_w_constrained` を `have_2a` が無いときは `np.inf` としておくことで、
`min(np.inf, w_r)` が自動的に `w_r` 側を選ぶため、式を分岐させずに
「stage2aが全滅していてもstage2bの異なるクラス解を使う」を満たせる。

非 relay-bp decoder（`converged1`/`converged2a_i`/`converged_r` が常に `True`）では
この分岐は自然に既存の case a/b/c の式へ簡約され、出力は変更前とビット単位で一致する。

### `ar-pec` / `ar-lec`（`ArgumentReweightingDecoder`）

- round0（stage1相当）が非収束の場合: `accept=False` を返し、logical error を強制
  True にする。round0が非収束の時点で以降のラウンドは実行しない。
- round0 が収束した後、途中のラウンドが非収束、またはラウンド間の一致条件
  （`PEC`: correctionが完全一致 / `LEC`: logical classが一致）を満たさなくなった
  場合は、その時点で `accept=False` として打ち切る（既存動作のまま、非収束かどうかは
  相変わらず明示的にチェックしていない — correction/logical class の不一致を通じて
  間接的に検出される）。この場合、logical error への強制上書きは行わない。

## logical error判定への反映

各実装は `metrics["__is_logical_error"]` を、decoder が relay-bp のときのみ
`DecodingResult.metrics` に含める（非relay-bpでは何も追加しない）。
`execution/worker.py` 側では

```python
is_logical_error = is_logical_error | override
```

という「Trueにのみ作用するOR」でこれを合成しているため、通常の
`prediction != true_observable` 判定を壊すことはなく、stage1非収束時にのみ
強制的にlogical errorとして扱われる。

## decoder_statsとの関係

`linearize_logicalgap` / `forced_gap_ml` は元々 `get_detail_stat=True` で
`baseline_iteration` / `forced_iteration` 等の decoder_stats を出力できる
（relay-bpの場合は `result.iterations`、非収束時は `np.nan`）。今回の変更でこれと
`forced_unconverged_confidence_value` は同一ファイル内で両立するようになった
（以前の `_relay_bp.py` 実装にはdecoder_stats機構自体が無く、両立していなかった）。

`_forcing_degradation_test.py` の `RelayBpForcingStageRunner`
（`forcing_degradation_test` 指標専用の実装、本ドキュメントの対象外）は非収束時も
`result.iterations` の生値をそのまま格納する。relay_bpのRust実装上、この値は非収束時は
設定された `max_iter`（あるいはrelayの複数leg合計反復数）相当の実消化回数であり、`-1`
などのセンチネルではない。`linearize_logicalgap`/`forced_gap_ml` 側は非収束時に
明示的に `np.nan` を格納する点で異なる（同じ「iteration数」という名前でも非収束時の
扱いが非対称なので、将来的にどちらかに揃えるかどうかは別途検討が必要）。

## テスト

`tests/test_relay_bp_nonconvergence.py` に、`RelayBpDecoderAdapter` を実クラスのまま
使い、内部の `_decoder`（relay_bp Rust decoder相当）だけをスタブ化したフェイクを使った
単体テストがある（`_rebuild()` を no-op に差し替えることで実 relay_bp 依存を回避）。
カバーしているシナリオ:

- stage1非収束 → 各指標で規定のconfidence値、`is_logical_error`強制True、stage2が
  一度も呼ばれないこと。
- stage2全滅 → `forced_unconverged_confidence_value` の `positive`/`negative`
  それぞれで規定値になり、`is_logical_error`は上書きされないこと。
- `reweighted_linearized_gap`: stage2aが全滅・stage2bのみ異なるlogical classに
  収束するケースで、フォールバックにならず既存のgap式で計算されること（回帰テスト）。
- `reweighted_linearized_gap`: stage2aが全滅・stage2bが同じlogical classに収束した
  場合は「全滅」扱いになり、config値が使われること。
- `ar-pec`/`ar-lec`: round0非収束時に `accept=False` かつ `is_logical_error` が
  強制されること。
- `metric_options.forced_unconverged_confidence_value` が未指定・不正値の場合に
  `ValueError` になること。
