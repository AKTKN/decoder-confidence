# Simulation package for decoder confidence project

## Simulation steps
1. Generate quantum circuit (`.stim`)
2. Create dem and sample it, then store them.
3. Run decode, calculate decoder confidence (logical gap, cluster llr, AR etc...)
4. Plot metrics distribution, post-selection performance. 

`qec-decoder-sim` is useful for generating quantum circuits of basic codes (e.g. surface code, superdense color code, BB code).

If you want to get exact logical gap for arbitrary quantum code, use `ILP-decoder`. This decoder supports exact logical gap calculation. Note that this method takes so long time compared to other decoder run(like BP, MWPM), so take care running shots and num of cpu cores. 

改めて、どういう図面を挿入するか整理。





まず目的は、forced gapが復号信頼度として機能するのかどうか（exact gapに近いかというよりより、本質的に重要な問題はまずこれ）。よって、横軸gap、縦軸に条件付き論理エラー率のplotを見せる。ILPとforcedの両方を提示。



Post-selection性能の図面



ExactとForcedのgap散布図（これphenomenologicalだとgapの値が離散的になるため、頻度も情報として含めたい場合どういう図面が良いか？）



（）