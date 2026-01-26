# CYC+

## Environment
```.bash
conda env create -f environment.yml
conda activate cyc
```

## Dataset


## Exe
```.bash
python3 generate.py --dir Base_Input
python3 main.py --input_dir Base_Input --output_dir Base_Output
python3 plot.py --input_dir Base_Input --output_dir Base_Output
```