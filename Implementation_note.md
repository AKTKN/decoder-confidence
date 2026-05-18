## To do

- demからcheckmatrixを作る際、hyperedgeをedge-mechanismにdecomposeするかをoptionで選択できるようにする必要がある。例えば、表面符号の場合、XYZ-decodingだとmatchingとの比較ができない。
    - dem をsampleする時点で分ける
- configにXYZ decodingをするかのoptionを追加(XZseparateする場合は、あらかじめ、元のcircuitからX basisのdetectorを取り除く必要がある。filter_by_basis関数でdecoder instance生成の前にあらかじめ処理)
    - basisを取り除けば、そのcircuitから生成したdemは既にdecomposeされる。表面符号の場合、この時点でhyperedge mechanismはなくなるので、dem_to_check_matrix関数でedge_matrixとかいちいち使う必要がなくなる。


## Future work
- implement analysis modules
- Downgrade variable type (e.g. use float 32 for metric instead float64) to reduce memory consumption.
- add cluster metrics support (by BP-LSD from `ldpc`)
- add heuristic logical gap calculation(by ensemble)
- adaptive confidence calculation


## Analysis Module
I want to analyze performance of confidence metrics. I'll create analysis-only modules in  `analysis/src`.
What I want to do is:
    1. Plot distribution of metrics(which has numerical value)
    2. Plot metric conditional logical error rate(and also do fitting). 
    3. Plot post-selection curve abort rate vs post-logical error rate.


### Implementation details
First, we need to load simulation data. The directory structure of simulation data is:
`result_dir_root/{circuit information}/decoding_result/{Simulation data for each decoder, metrics etc...}`.

We want to handle each simulation data with property, for example:
    - circuit information
    - decoder name
    - metric name
    - metric options
    - decoder options  etc...
I want to access these datas in analysis process, so define data class for each raw simulation data. I'm thinking which implementation is good. 
How about this? First, construct dataclass
Dataclass
 - raw data(from .parquet), metric data and is_logical_error data.
 - property data like above for each raw data.

then, we assign key for access(use defined with data dir) and combine them like {'logicalgap_ilp': Dataclass_object(including dir?)}.

and, the important point is that all metric is not numeric, for example, argument reweighting returns bool(accept/reject), so it cannot be plotted like logical gap which has numeric value(x-axis: value, y-axis: frequency), so we want to define some data structure for each metrics beforehand.

After implementing this, we will start build detail plotting modules.



