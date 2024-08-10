# export CUDA_VISIBLE_DEVICES=7
# export CUDA_LAUNCH_BLOCKING=1

dset_name=charades
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_cha/tgt_cc_crop
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
clip_length=1
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
bsz=8


gpunum=0

seed=2018

list="2025 2024"

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
--exp_id base_5_11_19_crop_${seed} \
--device ${device} \
--span_loss_type ${span_loss_type} \
--lr ${lr} \
--enc_layers ${enc_layers} \
--sim_loss_coef ${sim_loss_coef} \
--neg_loss_coef ${neg_loss_coef} \
--seed ${seed} \
--lr_gamma ${lr_gamma} \
--clip_length ${clip_length} \
--neg_choose_epoch ${neg_choose_epoch} \
--lr_drop ${lr_drop} \
--n_epoch 200 \
--max_v_l ${max_v} \
--dec_layers ${dec_layers} \
--num_queries 10 \
--m_classes "[5, 11.5, 19, 10000]" \
--cc_matching \
--tgt_embed \
--crop \
--fore_min 10 \
--back_min 10 \
--mid_min 10 \
--crop_random \
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
# --exp_id base_5_11_19_crop_all_${seed} \
# --device ${device} \
# --span_loss_type ${span_loss_type} \
# --lr ${lr} \
# --enc_layers ${enc_layers} \
# --sim_loss_coef ${sim_loss_coef} \
# --neg_loss_coef ${neg_loss_coef} \
# --seed ${seed} \
# --lr_gamma ${lr_gamma} \
# --clip_length ${clip_length} \
# --neg_choose_epoch ${neg_choose_epoch} \
# --lr_drop ${lr_drop} \
# --n_epoch 200 \
# --max_v_l ${max_v} \
# --dec_layers ${dec_layers} \
# --num_queries 10 \
# --m_classes "[5, 11.5, 19, 10000]" \
# --cc_matching \
# --tgt_embed \
# --crop \
# --fore_min 10 \
# --back_min 10 \
# --mid_min 10 \
# --crop_random \
# --crop_all \
# ${@:1}

done




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
# --exp_id base_5_14_${seed} \
# --device ${device} \
# --span_loss_type ${span_loss_type} \
# --lr ${lr} \
# --enc_layers ${enc_layers} \
# --sim_loss_coef ${sim_loss_coef} \
# --neg_loss_coef ${neg_loss_coef} \
# --seed ${seed} \
# --lr_gamma ${lr_gamma} \
# --clip_length ${clip_length} \
# --neg_choose_epoch ${neg_choose_epoch} \
# --lr_drop ${lr_drop} \
# --n_epoch 200 \
# --max_v_l ${max_v} \
# --dec_layers ${dec_layers} \
# --num_queries 10 \
# --m_classes "[5.67, 14, 10000]" \
# --cc_matching \
# --tgt_embed \
# ${@:1}