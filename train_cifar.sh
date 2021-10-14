DATE='211012'
GPU=1

EXPERIMENT='split_cifar100'
TASK_NUM=10
SEED=0

#FT
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'finetuning' --tasknum $TASK_NUM --seed $SEED --conv-net

#EWC
LAMB=12000
CPR_BETA=0.5
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA

#SI
C=1
CPR_BETA=0.8
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C --cpr-beta $CPR_BETA

#MAS
LAMB=3
CPR_BETA=0.5
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA

#RWalk
LAMB=8
CPR_BETA=0.9
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA 

#AGS-CL
LAMB=400
MU=10
RHO=0.3
CPR_BETA=0.4
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU --rho $RHO --cpr-beta $CPR_BETA

###############################################################################################################################

EXPERIMENT='split_cifar10_100'
TASK_NUM=11
SEED=0

#FT
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'finetuning' --tasknum $TASK_NUM --seed $SEED --conv-net

#EWC
LAMB=25000
CPR_BETA=0.4
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA

#SI
C=0.9
CPR_BETA=0.2
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C --cpr-beta $CPR_BETA

#MAS
LAMB=1
CPR_BETA=0.2
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA

#RWalk
LAMB=4
CPR_BETA=0.4
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA 

#AGS-CL
LAMB=7000
MU=20
RHO=0.2
CPR_BETA=0.1
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU --rho $RHO
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU --rho $RHO --cpr-beta $CPR_BETA

###############################################################################################################################

EXPERIMENT='split_cifar100_10'
TASK_NUM=11
SEED=0

#FT
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'finetuning' --tasknum $TASK_NUM --seed $SEED --conv-net

#EWC
LAMB=20000
CPR_BETA=0.6
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA

#SI
C=2
CPR_BETA=0.5
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C --cpr-beta $CPR_BETA

#MAS
LAMB=2
CPR_BETA=0.4
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA

#RWalk
LAMB=10
CPR_BETA=0.8
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA 

#AGS-CL
LAMB=8000
MU=10
RHO=0.3
CPR_BETA=1.0
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU --rho $RHO
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU --rho $RHO --cpr-beta $CPR_BETA

###############################################################################################################################

EXPERIMENT='split_cifar50_10_50'
TASK_NUM=11
SEED=0

#FT
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'finetuning' --tasknum $TASK_NUM --seed $SEED --conv-net

#EWC
LAMB=12000
CPR_BETA=0.8
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ewc_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA

#SI
C=2
CPR_BETA=0.9
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'si_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --c $C --cpr-beta $CPR_BETA

#MAS
LAMB=2
CPR_BETA=0.1
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'mas_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA

#RWalk
LAMB=10
CPR_BETA=0.6
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'rwalk_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --cpr-beta $CPR_BETA 

#AGS-CL
LAMB=9000
MU=10
RHO=0.3
CPR_BETA=0.5
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU --rho $RHO
CUDA_VISIBLE_DEVICES=$GPU python main.py --date $DATE --experiment $EXPERIMENT --approach 'ags_cl_cpr' --tasknum $TASK_NUM --seed $SEED --conv-net --lamb $LAMB --mu $MU --rho $RHO --cpr-beta 0.1