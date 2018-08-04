import sys
import os
import pickle
import subprocess

def get_best_dropout(model_name, lr, bsize):
    fin = open('test_result/epoch_dict.pkl', 'rb')
    d = pickle.load(fin)
    max_auc = 0
    best_dropout = 0
    for key in d:
        if key[0]==model_name and key[2]==lr and key[3]==bsize:
            if d[key]>max_auc:
                max_auc = d[key]
                best_dropout = key[1]
    return best_dropout

def get_best_lr(model_name, dropout, bsize):
    fin = open('test_result/epoch_dict.pkl', 'rb')
    d = pickle.load(fin)
    max_auc = 0
    best_lr = 0
    for key in d:
        if key[0]==model_name and key[1]==dropout and key[3]==bsize:
            if d[key]>max_auc:
                max_auc = d[key]
                best_lr = key[2]
    return best_lr

def get_best_bsize(model_name, dropout, lr):
    fin = open('test_result/epoch_dict.pkl', 'rb')
    d = pickle.load(fin)
    max_auc = 0
    best_bsize = 0
    for key in d:
        if key[0]==model_name and key[1]==dropout and key[2]==lr:
            if d[key]>max_auc:
                max_auc = d[key]
                best_bsize = key[3]
    return best_bsize

dropout_list = [0.5, 0.4, 0.3, 0.2, 0.1]
lr_list = [1, 0.25, 0.125, 0.0625]
bsize_list = [640, 320, 80]

model_name = sys.argv[1]
devices = sys.argv[2]

default_lr = 0.5
default_bsize = 160

for dropout in dropout_list:
    cmd1 = 'CUDA_VISIBLE_DEVICES=%s python train.py --model_name %s --learning_rate %s --drop_prob %s --batch_size %s --random_seed 1234'\
        % (devices, model_name, default_lr, dropout, default_bsize)
    cmd2 = 'CUDA_VISIBLE_DEVICES=%s python dev.py --model_name %s --learning_rate %s --drop_prob %s --batch_size %s --random_seed 1234'\
        % (devices, model_name, default_lr, dropout, default_bsize)
    print(cmd1)
    subprocess.call(cmd1,shell=True)
    print(cmd2)
    subprocess.call(cmd2,shell=True)

best_dropout = get_best_dropout(model_name, default_lr, default_bsize)

for lr in lr_list:
    cmd1 = 'CUDA_VISIBLE_DEVICES=%s python train.py --model_name %s --learning_rate %s --drop_prob %s --batch_size %s --random_seed 1234'\
        % (devices, model_name, lr, best_dropout, default_bsize)
    cmd2 = 'CUDA_VISIBLE_DEVICES=%s python dev.py --model_name %s --learning_rate %s --drop_prob %s --batch_size %s --random_seed 1234'\
        % (devices, model_name, lr, best_dropout, default_bsize)
    print(cmd1)
    subprocess.call(cmd1,shell=True)
    print(cmd2)
    subprocess.call(cmd2,shell=True)

best_lr = get_best_lr(model_name, best_dropout, default_bsize)

for bsize in bsize_list:
    cmd1 = 'CUDA_VISIBLE_DEVICES=%s python train.py --model_name %s --learning_rate %s --drop_prob %s --batch_size %s --random_seed 1234'\
        % (devices, model_name, best_lr, best_dropout, bsize)
    cmd2 = 'CUDA_VISIBLE_DEVICES=%s python dev.py --model_name %s --learning_rate %s --drop_prob %s --batch_size %s --random_seed 1234'\
        % (devices, model_name, best_lr, best_dropout, bsize)
    print(cmd1)
    subprocess.call(cmd1,shell=True)
    print(cmd2)
    subprocess.call(cmd2,shell=True)

best_bsize = get_best_bsize(model_name, best_dropout, best_lr)

print('====TUNING COMPLETED!====')
print('Best Param: Dropout = %s, Learning Rate = %s, Batch Size = %s' % (str(best_dropout), str(best_lr), str(best_bsize)))
