
## install conda environments 

```
conda env create --file leap.yml 
```
## train LEAP 

```
conda activate leap

```
## move to experiments folder
```
cd experiments/

```
## running shell file at /AAAI_LEAP/experiments/

## run code mujoco_0.3_final.sh

```
sh mujoco_0.3_final.sh 
```

## BASELINES MUJOCO

```
python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity '' --epoch 2 --model ncde_forecasting
python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity 'True' --epoch 2 --model decay_forecasting
python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity 'True' --epoch 2 --model odernn_forecasting
python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity 'True' --epoch 2 --model dt_forecasting
python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity '' --epoch 2 --model gruode_forecasting
```

## run code uea.sh

```
sh uea.sh 
```

## BASELINES UEA

```
python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --missing_rate 0.3 --model='ncde'
python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --missing_rate 0.3 --model='gruode'

```

## BASELINES UEA 
# NEEDS data preprocessing. (Due to the submission memory limitation)
# When you run this code, it would start from preprocessing step (it may take a while.).

```
python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --intensity 'True' --missing_rate 0.3 --model='dt'
python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --intensity 'True' --missing_rate 0.3 --model='decay'
python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --intensity 'True' --missing_rate 0.3 --model='odernn'
```
