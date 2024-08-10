# export CUDA_VISIBLE_DEVICES=7
# export CUDA_LAUNCH_BLOCKING=1

dset_name=charades_vgg
ctx_mode=video_tef
v_feat_types=vgg
t_feat_type=clip 
results_root=results_cha_vgg/base
device=1
enc_layers=3
dec_layers=3
query_num=30

span_loss_type=l1
sim_loss_coef=1
neg_loss_coef=0.5
exp_id=test
seed=2018
lr=1e-4
lr_gamma=0.1
clip_length=0.166666
neg_choose_epoch=70
lr_drop=80
max_v=75

######## data paths
train_path=data/charades/charades_sta_train_tvr_format.jsonl
eval_path=data/charades/charades_sta_test_tvr_format.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features/charades

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"vgg"* ]]; then
  v_feat_dirs+=(${feat_root}/vgg_features/rgb_features)
  (( v_feat_dim += 4096 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi


# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=300
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=8


gpunum=1

seed=2018

list="2025"

for seed in $list
do
  echo $seed

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python uvcom/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id base_${seed}_cta1.5 \
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--num_queries ${query_num} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--seed ${seed} \
--lr_gamma ${lr_gamma} \
--clip_length 1 \
--neg_choose_epoch ${neg_choose_epoch} \
--lr_drop ${lr_drop} \
--n_epoch 200 \
--max_v_l ${max_v} \
--dec_layers ${dec_layers} \
--cta_coef 1.5 \
${@:1}

# CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python uvcom/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id base_${seed}_cl_016 \
# --device ${device} \
# --span_loss_type ${span_loss_type} \
# --lr ${lr} \
# --num_queries ${query_num} \
# --enc_layers ${enc_layers} \
# --sim_loss_coef ${sim_loss_coef} \
# --neg_loss_coef ${neg_loss_coef} \
# --seed ${seed} \
# --lr_gamma ${lr_gamma} \
# --clip_length 0.166666 \
# --neg_choose_epoch ${neg_choose_epoch} \
# --lr_drop ${lr_drop} \
# --n_epoch 200 \
# --max_v_l ${max_v} \
# --dec_layers ${dec_layers}\
# ${@:1}


done