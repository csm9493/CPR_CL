DATE='211012'
GPU=3

EXPERIMENT='split_CUB200_new'
TASK_NUM=10
SEED=0

#EWC
LAMB=300000
CPR_BETA=0.4
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --lamb $LAMB --cpr-beta $CPR_BETA

#SI
C=50
CPR_BETA=0.6
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --c $C
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --c $C --cpr-beta $CPR_BETA

#MAS
LAMB=50
CPR_BETA=0.6
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --lamb $LAMB --cpr-beta $CPR_BETA

#RWalk
LAMB=300
CPR_BETA=0.9
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --lamb $LAMB --cpr-beta $CPR_BETA 

#AGS-CL
LAMB=2100
MU=0.5
RHO=0.1
CPR_BETA=0.7
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --lamb $LAMB --mu $MU --rho $RHO
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --model 'Resnet18' --batch-size 32 --lr 0.0005  --nepochs 50 --lamb $LAMB --mu $MU --rho $RHO --cpr-beta $CPR_BETA

