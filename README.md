# PPIformer demo for ICLR2024 rebuttal

Please note that this code is only intended for the examination by reviewers at ICLR2024. The repository will be siginificantly improved in future.

To install, run the following command from the root of the directory (it assumes having the `PPIRef` and `mutils` repositories on the same level with this one).
```
. scripts/installation/install.sh
```
You may also want to download the pre-trained models for ddG prediction from [zenodo](https://zenodo.org/records/10183718).


Then, to reproduce the test results from the paper, you can run
```
python scripts/test_ddg_regression.py
```
The script will make predictions with pre-trained ddG models and calculate the tables reported in our manuscript.
