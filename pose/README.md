COCO

python pose/test.py pose/config/hrnet_w48_coco_256x192.py data/unit_infer/checkpoint/pose/best_AP_epoch_90.pth --eval mAP

ETRI

python pose/test.py pose/config/hrnet_w48_etri_256x192.py data/unit_infer/checkpoint/pose/best_AP_epoch_90.pth --eval mAP --iscoco True
