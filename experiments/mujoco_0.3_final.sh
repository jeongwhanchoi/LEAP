for model in 'learnable_forecasting' 
do
    for missing_rate in 0.3
    do
        for seed in 118
        do
            for c1 in 0.0001 
            do
                for c2 in 0.0001
                do
                    for h_channel in 80 
                    do 
                        for hh_channel in 50
                        do 
                            for lr in  0.0001 
                            do
                                for time_seq in 50 
                                do
                                    for y_seq in 10 
                                    do
                                        for step_mode in 'valloss' 
                                        do
                                            CUDA_VISIBLE_DEVICES=0 python3 -u mujoco.py --seed $seed --model $model --h_channels $h_channel --hh_channels $hh_channel --layers 5 --lr $lr --c1 $c1 --c2 $c2 --method "rk4" --weight_decay 0 --step_mode $step_mode --missing_rate $missing_rate --time_seq $time_seq --y_seq $y_seq --intensity '' --epoch 2 > ../experiments/test/LEAP_y_{$y_seq}_{$model}.csv
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
done

# Baselines! 

# CUDA_VISIBLE_DEVICES=0 python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity '' --epoch 2 --model ncde_forecasting
# CUDA_VISIBLE_DEVICES=0 python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity 'True' --epoch 2 --model decay_forecasting
# CUDA_VISIBLE_DEVICES=0 python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity 'True' --epoch 2 --model odernn_forecasting
# CUDA_VISIBLE_DEVICES=0 python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity 'True' --epoch 2 --model dt_forecasting
# CUDA_VISIBLE_DEVICES=0 python3 -u mujoco.py --h_channels 80 --hh_channels 50 --layers 5 --lr 0.0001 --c1 0.0001 --c2 0.0001 --method "rk4" --weight_decay 0  --missing_rate 0.3 --time_seq 50 --y_seq 10 --intensity '' --epoch 2 --model gruode_forecasting


