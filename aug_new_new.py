from utils.basic_utils import save_jsonl, load_jsonl
from copy import deepcopy
import random
import numpy as np
from utils.length_aug import *



max_v_l = 1000000000
seed = 0
random.seed(seed)
np.random.seed(seed)

# org_datalist = load_jsonl('data/highlight_train_release.jsonl')
# clip_len = 2

# org_datalist = load_jsonl('data/tacos/train.jsonl')
# clip_len = 2

org_datalist = load_jsonl('data/charades/charades_sta_train_tvr_format.jsonl')
clip_len = 1


new_datalist = []


for data in org_datalist:

            new_datalist.append(deepcopy(data))

            ctx_l = int(data['duration'] // clip_len) if data['duration'] % clip_len == 0 else int(data['duration'] // clip_len) + 1
            # ctx_l = min(round(data['duration'] // clip_len), self.max_v_l)

            ###############################################
            # moment와 non-moment 구하기
            ###############################################

            if 'relevant_clip_ids' in data: # QVHighlights

                all_clips = np.zeros(ctx_l)
                all_clips[data['relevant_clip_ids']] = 1

                moments = find_ones_groups(all_clips, clip_len)
                assert moments == data['relevant_windows']

                non_moments = find_zeros_groups(all_clips, clip_len)

            else: # Charades, TACoS (single moment)
                moments = data['relevant_windows']
                non_moments = []
                if moments[0][0] != 0:
                    non_moments.append([0, moments[0][0]])
                if moments[0][1] != data['duration']:
                    non_moments.append([moments[0][1], data['duration']])    
                non_moments[-1][1] = ctx_l * clip_len   

            # 만약 non-moment가 없다면 이 data는 pass
            if not non_moments:
                continue 
            
            # crop augmentation
            new_crop_data = crop(data, moments=moments, non_moments=non_moments, thres_crop=10, ctx_l=ctx_l, clip_len=clip_len)
            if new_crop_data:
                new_datalist.append(new_crop_data)

            # merge augmentation for multi-moments dataset

            if 'saliency_scores' in data:
                new_merge_data = merge_multi_moments(data, moments=moments, non_moments=non_moments, thres_merge=10, ctx_l=ctx_l, clip_len=clip_len)

                if new_merge_data:
                    new_datalist.append(new_merge_data)

            else:
                s, e = data['relevant_windows'][0]
                rs = int(s // clip_len) if s % clip_len == 0 else int(s // clip_len) + 1
                re = int(e // clip_len)
                rs, re = rs * clip_len, re * clip_len

                moments = [[rs, re]]
                non_moments = []
                if rs - clip_len > 0:
                    non_moments.append([0, rs - clip_len])
                if re + clip_len < ctx_l * clip_len:
                    non_moments.append([re + clip_len, ctx_l * clip_len ])    

                new_merge_data = merge_single_moment(data, moments=moments, non_moments=non_moments, thres_merge=10, ctx_l=ctx_l, clip_len=clip_len)
                if new_merge_data:
                    new_datalist.append(new_merge_data)

print(f"Oracle Crop : {len(org_datalist)} -> {len(new_datalist)}")
save_jsonl(new_datalist, f'cha_new_new_aug.jsonl')
# save_jsonl(new_datalist, f'new_merge_tacos.jsonl')
