
for dataset_name in "CharacterTrajectories"
do
    for c1 in 1e-6
    do
        for c2 in 0
        do
            for lr in 0.001
            do
                for h_channel in 40
                do 
                    for hh_channel in 100
                    do
                        for layer in 3
                        do
                            for weight_decay in 0
                            do
                                for step_mode in 'valloss'
                                do
                                    for seed in 198
                                    do
                                        
                                        CUDA_VISIBLE_DEVICES=1 python3 -u uea.py  --dataset_name $dataset_name --seed $seed --model='learnable' --h_channels $h_channel --hh_channels $hh_channel --layer $layer --lr $lr --c1 $c1 --c2 $c2 --method "rk4" --weight_decay $weight_decay --step_mode $step_mode --missing_rate 0.3 > ../experiments/test/LEAP_UEA.csv
                                    done
                                done
                            done
                        done
                    done
                done    
            done
        done
    done
done

# BASELINES!
# CUDA_VISIBLE_DEVICES=1 python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --missing_rate 0.3 --model='learnable'
# CUDA_VISIBLE_DEVICES=1 python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --missing_rate 0.3 --model='ncde'
# CUDA_VISIBLE_DEVICES=1 python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --missing_rate 0.3 --model='gruode'
# CUDA_VISIBLE_DEVICES=1 python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --intensity 'True' --missing_rate 0.3 --model='dt'
# CUDA_VISIBLE_DEVICES=1 python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --intensity 'True' --missing_rate 0.3 --model='decay'
# CUDA_VISIBLE_DEVICES=1 python3 -u uea.py --dataset_name CharacterTrajectories --h_channels 40 --hh_channels 100 --layer 3 --lr 0.001 --c1 1e-6 --c2 0 --method "rk4" --weight_decay 0  --intensity 'True' --missing_rate 0.3 --model='odernn'

