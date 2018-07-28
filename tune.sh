model="cnn_ave"
devices="0"

for lr in 0.5 0.1 0.01 0.001 0.0001
do
    for dropout in 0.5 0.4 0.3 0.2 0.1
    do
        echo "learning rate = $lr"
        echo "dropout = $dropout"
        CUDA_VISIBLE_DEVICES=$devices python train.py --model_name $model --learning_rate $lr --drop_prob $dropout
        CUDA_VISIBLE_DEVICES=$devices python dev.py --model_name $model --learning_rate $lr --drop_prob $dropout
    done
done

echo "tuning for $model finished!"