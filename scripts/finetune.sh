# conda activate ssl_env
# python downstream_task_evaluation.py data=wisdm_10s report_root=~/ssl-wearables/data/reports/ evaluation.flip_net_path=/data/ssl_wearable/model/mtl_best.mdl evaluation=all
# python downstream_task_evaluation.py data=mymove_5s report_root=~/ssl-wearables/data/reports/ evaluation=all5s
python downstream_task_evaluation.py data=mymove_10s report_root=~/ssl-wearables/data/reports/ evaluation=all