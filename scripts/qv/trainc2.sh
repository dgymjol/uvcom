# export CUDA_VISIBLE_DEVICES=7
# export CUDA_LAUNCH_BLOCKING=1

dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_13_19
device=1
enc_layers=3
dec_layers=3
query_num=10
n_txt_mu=5
n_visual_mu=30

span_loss_type=l1
sim_loss_coef=1
neg_loss_coef=0.5
exp_id=test
seed=2023
lr=1e-4
lr_gamma=0.1
neg_choose_epoch=80
lr_drop=100

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32


gpunum=2

# seed=2018

list="2021 2022 2023 2024 2025"

results_root=results_12_36_65_crop

seed=2023

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
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--lr_gamma ${lr_gamma} \
--dec_layers ${dec_layers} \
--lr_drop ${lr_drop} \
--em_iter 5 \
--n_txt_mu ${n_txt_mu} \
--n_visual_mu ${n_visual_mu} \
--neg_choose_epoch ${neg_choose_epoch} \
--num_queries 10 \
--exp_id base_crop_${seed} \
--m_classes "[12, 36, 65, 150]" \
--no_text \
--tgt_embed \
--cc_matching \
--seed ${seed} \
--crop \
--fore_min 10 \
--back_min 10 \
--mid_min 10 \
--crop_random \
${@:1}

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
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--lr_gamma ${lr_gamma} \
--dec_layers ${dec_layers} \
--lr_drop ${lr_drop} \
--em_iter 5 \
--n_txt_mu ${n_txt_mu} \
--n_visual_mu ${n_visual_mu} \
--neg_choose_epoch ${neg_choose_epoch} \
--num_queries 10 \
--exp_id base_crop_all_${seed} \
--m_classes "[12, 36, 65, 150]" \
--no_text \
--tgt_embed \
--cc_matching \
--seed ${seed} \
--crop \
--fore_min 10 \
--back_min 10 \
--mid_min 10 \
--crop_random \
--crop_all \
${@:1}


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
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--lr_gamma ${lr_gamma} \
--dec_layers ${dec_layers} \
--lr_drop ${lr_drop} \
--em_iter 5 \
--n_txt_mu ${n_txt_mu} \
--n_visual_mu ${n_visual_mu} \
--neg_choose_epoch ${neg_choose_epoch} \
--num_queries 10 \
--exp_id base_crop_merge_${seed} \
--m_classes "[12, 36, 65, 150]" \
--no_text \
--tgt_embed \
--cc_matching \
--seed ${seed} \
--crop \
--fore_min 10 \
--back_min 10 \
--mid_min 10 \
--crop_random \
--merge \
${@:1}


seed=2021
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
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--lr_gamma ${lr_gamma} \
--dec_layers ${dec_layers} \
--lr_drop ${lr_drop} \
--em_iter 5 \
--n_txt_mu ${n_txt_mu} \
--n_visual_mu ${n_visual_mu} \
--neg_choose_epoch ${neg_choose_epoch} \
--num_queries 10 \
--exp_id base_crop_merge_${seed} \
--m_classes "[12, 36, 65, 150]" \
--no_text \
--tgt_embed \
--cc_matching \
--seed ${seed} \
--crop \
--fore_min 10 \
--back_min 10 \
--mid_min 10 \
--crop_random \
--merge \
${@:1}

seed=2023

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
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--lr_gamma ${lr_gamma} \
--dec_layers ${dec_layers} \
--lr_drop ${lr_drop} \
--em_iter 5 \
--n_txt_mu ${n_txt_mu} \
--n_visual_mu ${n_visual_mu} \
--neg_choose_epoch ${neg_choose_epoch} \
--num_queries 10 \
--exp_id base_crop_all_merge_${seed} \
--m_classes "[12, 36, 65, 150]" \
--no_text \
--tgt_embed \
--cc_matching \
--seed ${seed} \
--crop \
--fore_min 10 \
--back_min 10 \
--mid_min 10 \
--crop_random \
--merge \
--crop_all \
${@:1}