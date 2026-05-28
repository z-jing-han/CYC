# CYC+

## Environment

### Test one
```.bash
./run.sh
./run.sh [Input_dir] [Output_dir]
```

### Test different V
```
./V_test_exp.sh [Base_Input_dir] [Base_Output_dir] [start exp V] [start digit V] [end exp V] [start digit V]
./V_test_exp.sh -d
```

### Notation Mapping
|Formal Algorithm Name|Config Run List Name|Algorithm Class|Decoder Class|file name|
|-|-|-|-|-|
|DWPA|"DWPA"|DWPASolver|Nan|`dwpa_solver/dwpa.py`|
|LF|"DWPALF"|DWPASolver|Nan|`dwpa_solver/dwpa.py`|
|VO|"DWPAVO"|DWPASolver|Nan|`dwpa_solver/dwpa.py`|
|HF|"DWPAHF"|DWPASolver|Nan|`dwpa_solver/dwpa.py`|
|Lya-OPT|"AO"|AOSolver|Nan|`dwpa_opt/AO.py`|
|DARLA|"MAAOPPO*", "MARAOPPO*"|MAAOPPOSolver|AOXT, AOXP|`marl_related/algorithm_relate/ao_solver.py`|
|RMAPPO|"MARPPO_XT_CTDE"|MARPPOSolver|XT|`marl_related/algorithm_relate/rmappo_solver.py`|


"6-3": ["DWPA", "AO", "MARPPO_XT_CTDE"]
    DWPA              -> DWPA
    AO                -> Lya-OPT
    MARPPO_XT_CTDE    -> RMAPPO
"6-5-1": ["MARPPO_XT_CTDE", "MAPPO_XT_CTDE"]
    MARPPO_XT_CTDE    -> RMAPPO
    MAPPO_XT_CTDE     -> MAPPO
"6-5-2": ["MARPPO_XT_CTDE", "MARPPO_XP_CTDE"]
    MARPPO_XT_CTDE    -> RMAPPO-XT
    MARPPO_XP_CTDE    -> RMAPPO-XP
"6-5-3": ["MARPPO_XT_CTDE", "MAAOPPO_AOXT_CTDE"]
    MARPPO_XT_CTDE    -> RMAPPO
    MAAOPPO_AOXT_CTDE -> DecouplePPO

### Current Algorithm run list

```.json
"run_list": [
    "AO", "DWPA", "DWPALF", "DWPAVO", "DWPAHF", "FIXTIME", "GUROBI", "DOLA22", "ICSOC19", "YCL24",
    "MAAOPPO_AOXP_Dec", "MAAOPPO_AOXP_CTDE", "MAAOPPO_AOXT_Dec", "MAAOPPO_AOXT_CTDE",
    "MAPPO_XP_Dec", "MAPPO_XP_CTDE", "MAPPO_XT_Dec", "MAPPO_XT_CTDE",
    "MATWOPPO_AOXP_Dec", "MATWOPPO_AOXP_CTDE", "MATWOPPO_AOXT_Dec", "MATWOPPO_AOXT_CTDE",
    "MARAOPPO_AOXP_Dec", "MARAOPPO_AOXP_CTDE", "MARAOPPO_AOXT_Dec", "MARAOPPO_AOXT_CTDE",
    "MARPPO_XP_Dec", "MARPPO_XP_CTDE", "MARPPO_XT_Dec", "MARPPO_XT_CTDE",
    "MARTWOPPO_AOXP_Dec", "MARTWOPPO_AOXP_CTDE", "MARTWOPPO_AOXT_Dec", "MARTWOPPO_AOXT_CTDE"
],
"plot_groups": {
    "dwpa": ["DWPA", "DWPALF", "DWPAVO", "DWPAHF", "FIXTIME", "AO"],
    "competitor": ["DWPA", "DOLA22", "YCL24", "ICSOC19"],
    "MAAO": ["AO", "MAAOPPO_AOXP_Dec", "MAAOPPO_AOXP_CTDE", "MAAOPPO_AOXT_Dec", "MAAOPPO_AOXT_CTDE"],
    "MA": ["AO", "MAPPO_XP_Dec", "MAPPO_XP_CTDE", "MAPPO_XT_Dec", "MAPPO_XT_CTDE"],
    "MATWO": ["AO", "MATWOPPO_AOXP_Dec", "MATWOPPO_AOXP_CTDE", "MATWOPPO_AOXT_Dec", "MATWOPPO_AOXT_CTDE"],
    "MARAO": ["AO", "MARAOPPO_AOXP_Dec", "MARAOPPO_AOXP_CTDE", "MARAOPPO_AOXT_Dec", "MARAOPPO_AOXT_CTDE"],
    "MAR": ["AO", "MARPPO_XP_Dec", "MARPPO_XP_CTDE", "MARPPO_XT_Dec", "MARPPO_XT_CTDE"],
    "MARTWO": ["AO", "MARTWOPPO_AOXP_Dec", "MARTWOPPO_AOXP_CTDE", "MARTWOPPO_AOXT_Dec", "MARTWOPPO_AOXT_CTDE"],
    "BEST": ["AO", "MARPPO_XT_CTDE", "MAAOPPO_AOXT_CTDE", "MARAOPPO_AOXT_CTDE"]
}
```

#### MARL Config Run list Naming Convention
- Divided into three parts: Algorithm, Decoder, and [CTDE|Dec]
  - Assembled under `marl_related/runner.py`
- Algorithm, located in `marl_related/algorithm_related/`
  - Starts with "MA"
    - "MAPPO": All actions are determined by the Agent
      - `mappo.py`
    - "MAAOPPO": Computing uses an analytical solution, Offloading uses PPO
      - `ao_solver.py`
    - "MATWOPPO": Computing and Offloading use independent PPOs
      - `split_solver.py`
  - Starts with "MAR", completely symmetric to the "MA" methods, but the Network is replaced with RNN
    - "MARPPO"
      - `rmappo.py`
    - "MARAOPPO"
      - `rmappo.py`
    - "MARTWOPPO"
      - `rmappo.py`
- Decoder, located in `marl/action_related`
  - Starts with "AO": Indicates learning offloading only
    - "AOXP": Determines Task Size and Transmission Power
      - `ao_decoders.py`
    - "AOXT": Determines Task Size and Transmission Time
      - `ao_decoders.py`
  - Does not start with "AO": Learning includes both Computing and Offloading
    - "XP": Same above
      - `decoders.py`
    - "XT": Same above
      - `decoders.py`
  - FrequencyDecoder, QueueDecoder
    - Can only be used by the Computing Agent in "TWO", representing the frequency and the queue ratio to be consumed, respectively
      - `split_decoders.py`
- Training Methods
  - CTDE: Centralized training
  - "Dec": Each Agent operates independently; other Agents are considered part of the environment