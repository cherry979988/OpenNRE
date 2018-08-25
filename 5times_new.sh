devices=$1
model=$2
dropout=$3
lr=$4
bsize=$5

for i in 1 2 3 4 5
do
	echo "Run #$i"
	CUDA_VISIBLE_DEVICES=$devices python train.py --model_name $model --drop_prob $dropout --learning_rate $lr --batch_size $bsize --random_seed $i
	CUDA_VISIBLE_DEVICES=$devices python dev.py --model_name $model --drop_prob $dropout --learning_rate $lr --batch_size $bsize --random_seed $i
	CUDA_VISIBLE_DEVICES=$devices python test.py --model_name $model --drop_prob $dropout --learning_rate $lr --batch_size $bsize --random_seed $i
	python modify_name.py $model $i $lr $dropout $bsize
done
