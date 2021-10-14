DATE='211012'
GPU=2

EXPERIMENT='omniglot'
TASK_NUM=50
SEED=0

#FT
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'finetuning' --tasknum $TASK_NUM --seed $SEED --conv-net

EWC
LAMB=100000
CPR_BETA=1.0
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA 

#SI
C=8
CPR_BETA=0.7
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C &
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C --cpr-beta $CPR_BETA

#MAS
LAMB=10
CPR_BETA=0.6
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA

#RWalk
LAMB=3000
CPR_BETA=0.6
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA 

#AGS-CL
LAMB=1000
MU=7
RHO=0.5
CPR_BETA=1.0
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU --rho $RHO
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU --rho $RHO --cpr-beta $CPR_BETA