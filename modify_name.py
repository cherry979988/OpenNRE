import os,sys

model = sys.argv[1]
stamp = int(sys.argv[2])
lr = float(sys.argv[3])
dropout = float(sys.argv[4])
bsize = int(sys.argv[5])

filein = 'test_result/' + model + '_' + str(dropout) + '_' + str(lr) + '_x_test.npy'
fileout = 'test_result/' + model + '_' + str(dropout) + '_' + str(lr) + '_x_test_' + str(stamp) + '.npy'

os.rename(filein, fileout)

filein = 'test_result/' + model + '_' + str(dropout) + '_' + str(lr) + '_y_test.npy'
fileout = 'test_result/' + model + '_' + str(dropout) + '_' + str(lr) + '_y_test_' + str(stamp) + '.npy'

os.rename(filein, fileout)
