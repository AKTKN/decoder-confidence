# relay-bp: 非収束時のconfidence指標・logical error判定の挙動

このドキュメントは `src/decoder_confidence/decoding/_relay_bp.py`
（`RelayBpMetricDecoder`）が、decoder に `relay-bp` を使う場合に、BP が収束しなかった
ケースをどう扱うかを説明する。対象となる指標(`metric`)は次の4つ:

- `linearized_logicalgap`
- `forced_gap_ml`
- `reweighted_linearized_gap`
- `argument_reweighting`（`test_type`/`criterion` により ar-pec / ar-lec 相当を切り替え）

`forcing_degradation_test`（`_forcing_degradation_test.py`）は別実装・別指標であり、
本ドキュメントの対象外。

## 背景

`RelayBpMetricDecoder` は各指標を「baseline decode (stage1) → 各observableを強制反転
させた forced decode (stage2)」という2段階構成で実装している。以前の実装では、stage1/
stage2 が収束しなかった場合に `FALLBACK_GAP = -32768.0` / `FALLBACK_FORCED_GAP = 0.0`
という決め打ちのセンチネル値を返すだけで、次の問題があった。

1. 非収束であることが `is_logical_error` の判定に反映されない。
2. 非収束時の confidence 値を実験設定から選べない。
3. `reweighted_linearized_gap` では stage2a（各observable強制反転のconstrained decode）
   が全滅した場合、stage2b（reweightされたunconstrained decode）が baseline と異なる
   logical class に収束していてもそれを無視してフォールバックしてしまうバグがあった。

本実装ではこれらを解消し、非収束時の挙動を要件通りに統一した。

## 2段階デコードの構成（おさらい）

- **stage1**（`_stage1`, `_relay_bp.py:187`）: 通常の（制約なしの）decodeを1回行う。
  `converged1, c1, l1, w1` を返す（`converged1` は `result.success`、`c1` は correction、
  `l1` は correction から求めた logical class、`w1` はその重み）。
- **stage2 (constrained, "2a")**（`_stage2_constrained`, `_relay_bp.py:196`）:
  observable `i` ごとに、パリティ検査行列へ observable の行を追加し、
  syndrome に `1 - l1[i]` を追加した上で再度decodeする。これにより、収束した場合は
  必ず `l1` と observable `i` のビットが異なる correction が得られる（構成上保証される）。
  `reweighted_linearized_gap` ではこれを "stage2a" と呼ぶ。
- **stage2b**（`reweighted_linearized_gap` のみ）: priorをbaselineのcorrectionに向けて
  再重み付けした上で、制約なしの decode をもう一度行う。stage2aと異なり、こちらは
  収束しても baseline と同じ logical class に落ち着く可能性がある。

## 非収束時の挙動まとめ

| ケース | linearized_logicalgap | forced_gap_ml | reweighted_linearized_gap | argument_reweighting |
|---|---|---|---|---|
| stage1 (baseline) が非収束 | `-np.inf` | `0.0` | `-np.inf` | `accepted=False` |
| stage1 非収束時の logical error | **強制 True** | **強制 True** | **強制 True** | **強制 True** |
| stage1 非収束時の stage2 実行 | 行わない | 行わない | 行わない | （ラウンド2以降を実行しない） |
| stage2 が実質的に全滅 | config値 | config値 | config値 | 途中ラウンドが非収束/不一致で `accepted=False`（既存動作のまま） |
| stage2 全滅時の logical error | 上書きしない（stage1の予測で判定） | 同左 | 同左 | 同左（`accepted=False` のみ、logical errorへの強制はなし） |
| stage2 に1つ以上、baselineと異なるlogical classの収束解がある | それを使って既存の式で計算 | 既存のML選択ロジックのまま | それを使って既存の式で計算（stage2a全滅でもstage2bが使える場合はそれを使う） | ラウンド間の一致判定（既存動作） |

「stage1非収束」と「stage2全滅+`negative`設定」はどちらも `-np.inf`（あるいは
`forced_gap_ml`なら`0.0`）になり、confidence値だけでは区別できない。これは意図した
仕様（要件確認済み）であり、区別用の補助カラムは追加していない。

## metric_options: `forced_unconverged_confidence_value`

`linearized_logicalgap` / `forced_gap_ml` / `reweighted_linearized_gap` の3指標では
`metric_options.forced_unconverged_confidence_value` が **必須** になった
（`_parse_relay_bp_metric_options`, `_relay_bp.py:80`）。値は `"positive"` または
`"negative"`（大文字小文字は無視、strip される）。未指定・不正な値は `ValueError`。

`argument_reweighting` はこのオプションを使わない（boolean の `accept`/`reject` のみで、
数値のconfidenceを持たないため）。

「stage2が実質的に全滅」した場合の値は次の通り:

| forced_unconverged_confidence_value | linearized_logicalgap | forced_gap_ml | reweighted_linearized_gap |
|---|---|---|---|
| `positive` | `np.inf` | `np.inf` | `np.inf` |
| `negative` | `-np.inf` | `0.0` | `-np.inf` |

yaml設定例:

```yaml
decoder: RELAY-BP
metric: linearized_logicalgap

decoder_options:
  set_max_iter: 30

metric_options:
  forced_unconverged_confidence_value: negative
```

`reweighted_linearized_gap` の場合は既存必須オプション `b` と併記する:

```yaml
metric: reweighted_linearized_gap
metric_options:
  b: 2.0
  forced_unconverged_confidence_value: negative
```

## 各指標の詳細

### `linearized_logicalgap` / `forced_gap_ml`（`_relay_bp.py:214`, `:248`）

```
stage1 非収束
  → pred = l1（非収束時のstage1予測をそのまま使う）, confidence = -inf(linearized) / 0(forced_gap_ml)
    logical error を強制 True、stage2 は実行しない

stage1 収束
  num_obs == 0、または stage2 の全observableが非収束
  → pred は変わらず、confidence = forced_unconverged_confidence_value に基づく値
    logical error への上書きはなし（stage1の予測 = l1 で判定される）

  stage2 に1つ以上収束したinstanceがある
  → 既存の計算式のまま（変更なし）
    - linearized_logicalgap: 収束した中の最小重み - w1
    - forced_gap_ml: [stage1解] + [収束したstage2解たち] をプールし、
      最小重みをML予測として採用（predがstage1と入れ替わることがある）
```

### `reweighted_linearized_gap`（`_relay_bp.py:286`）

stage2a（constrained、各observable強制反転）は収束すれば構成上必ず baseline と異なる
logical class になるが、stage2b（reweighted unconstrained）は収束しても baseline と
**同じ** logical class になり得る。そのため「stage2が実質的に全滅したか」の判定は
stage2a単独ではなく、次の条件で行う:

```python
have_2a = 収束したstage2aインスタンスが1つ以上ある
have_2b_diff = stage2bが収束 かつ baselineと異なるlogical class

if not have_2a and not have_2b_diff:
    # 要件2: どちらの経路からもbaselineと異なる収束解を得られなかった
    confidence = forced_unconverged_confidence_value に基づく値
    logical errorへの上書きなし
```

`have_2a` が False でも `have_2b_diff` が True であれば、既存のgap計算式
（`gap = min(best_w_constrained, w_r) - w1` など）はそのまま利用できる。
`best_w_constrained` を `have_2a` が無いときは `np.inf` としておくことで、
`min(np.inf, w_r)` が自動的に `w_r` 側を選ぶため、式を分岐させずに要件3
（stage2aが全滅していてもstage2bの異なるクラス解を使う）を満たせる。これが
今回修正した箇所（旧実装は `not converged_2a` の時点で即座にフォールバックしており、
stage2bの結果を一切見ていなかった）。

### `argument_reweighting`（`_relay_bp.py:358`）

- round0（stage1相当）が非収束の場合: `accepted=False` を返し、**今回追加で**
  logical error を強制 True にする（それ以外の3指標と同じ扱い）。round0が非収束の
  時点で以降のラウンドは実行しない。
- round0 が収束した後、途中のラウンドが非収束、またはラウンド間の一致条件
  （`PEC`: correctionが完全一致 / `LEC`: logical classが一致）を満たさなくなった
  場合は、その時点で `accepted=False` として打ち切る。この「途中で収束しない、または
  logical classが変わったかどうか判断できない場合は強制的にabort（=reject）とする」
  というルールは元々の実装がそのまま満たしていたため、変更していない。
  この場合、logical error への強制上書きは行わない（`accepted` はconfidenceの
  reject/acceptを表すだけで、logical errorの真偽はround0の予測とtrue observableの
  比較でそのまま決まる）。

## logical error判定への反映

各 `_shot_*` メソッドは `(pred, metric_value, obs_flip_idx, force_logical_error)` の
4要素を返すようになり、`decode()`（`_relay_bp.py:148`）が shot ごとに
`force_logical_error` を集約して、常に `metrics["__is_logical_error"]` として返す
(該当なしなら全て `False`)。

`execution/worker.py` 側では

```python
is_logical_error = is_logical_error | override
```

という「Trueにのみ作用するOR」でこれを合成しているため、通常の
`prediction != true_observable` 判定を壊すことはなく、stage1非収束時にのみ
強制的にlogical errorとして扱われる。

## 補足: BPのiteration回数とdecode_statsの非収束時の扱い

`_relay_bp.py` の `RelayBpMetricDecoder` にはBP iteration数を出力する
decoder_stats機構自体が存在しないため、今回の変更のスコープには含まれていない。
参考までに、他の経路での扱いを記録しておく:

- `_linearize_logicalgap.py:175` / `_forced_gap.py:153`
  （`LinearizeLogicalGapDecoder` / `ForcedGapMLDecoder` の `get_detail_stat=True`
  経路。BP-LSD/relay-bp汎用アダプタを使う実装）:
  `result.success` が False の場合、実際のiteration数を捨てて明示的に `np.nan` を
  格納している。
- `_forcing_degradation_test.py` の `RelayBpForcingStageRunner`
  （`forcing_degradation_test` 指標専用の実装）:
  非収束時も `result.iterations` の生値をそのまま格納する。relay_bpのRust実装上、
  この値は非収束時は設定された `max_iter`（あるいはrelayの複数leg合計反復数）相当の
  実消化回数であり、`-1` などのセンチネルではない。

この2経路は同じ「iteration数」という名前のカラムでも非収束時の扱いが非対称なので、
将来的にどちらかに揃えるかどうかは別途検討が必要。

## テスト

`tests/test_relay_bp_metric_decoder.py` に、`decode_detailed_single` の戻り値を
シーケンス制御できるフェイクアダプタを使った単体テストを追加した。カバーしている
シナリオ:

- stage1非収束 → 各指標で規定のconfidence値、`is_logical_error`強制True、stage2が
  一度も呼ばれないこと。
- stage2全滅 → `forced_unconverged_confidence_value` の `positive`/`negative`
  それぞれで規定値になり、`is_logical_error`は上書きされないこと。
- `reweighted_linearized_gap`: stage2aが全滅・stage2bのみ異なるlogical classに
  収束するケースで、フォールバックにならず既存のgap式で計算されること（バグ修正の
  回帰テスト）。
- `reweighted_linearized_gap`: stage2aが全滅・stage2bが同じlogical classに収束した
  場合は「全滅」扱いになり、config値が使われること。
- `argument_reweighting`: baseline非収束時に `accepted=False` かつ
  `is_logical_error` が強制されること。
- `metric_options.forced_unconverged_confidence_value` が未指定・不正値の場合に
  `ValueError` になること。
