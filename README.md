# CYC+

## Environment

### Test one
```.bash
./run.sh [Input_dir] [Output_dir]
```

### Test different V
```
./V_test_exp.sh [Base_Input_dir] [Base_Output_dir] [start exp V] [start digit V] [end exp V] [start digit V]
./V_test_exp.sh -d
```

### Data Size Parameter Note
|Time|On Size|Of size|Var Size|Average Size|Max Edge Queue|
|:-:|:-:|:-:|:-:|:-:|:-:|
|1sec|64MB|3.2MB|2MB|26MB|
|1hour||

### TODO
+ Find the suitable name style

### Current Algorithm run list

"run_list": [
    "AO", "DWPA", "DWPALF", "DWPAVO", "DWPAHF", "FIXTIME", "GUROBI", "DOLA22", "ICSOC19", "YCL24",
    "MAAOPPO_AOXP_Dec", "MAAOPPO_AOXP_CTDE", "MAAOPPO_AOXT_Dec", "MAAOPPO_AOXT_CTDE",
    "MAPPO_XP_Dec", "MAPPO_XP_CTDE", "MAPPO_XT_Dec", "MAPPO_XT_CTDE",
    "MATWOPPO_AOXP_Dec", "MATWOPPO_AOXP_CTDE", "MATWOPPO_AOXT_Dec", "MATWOPPO_AOXT_CTDE"
]
