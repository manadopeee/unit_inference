COCO

python pose/test.py pose/config/hrnet_w48_coco_256x192.py data/unit_infer/checkpoint/pose/pose.pth --eval mAP

option --class_anno_path class.json

ETRI

python pose/test.py pose/config/hrnet_w48_etri_256x192.py data/unit_infer/checkpoint/pose/pose.pth --eval mAP --iscoco True
