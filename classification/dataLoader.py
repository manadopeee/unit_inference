import torch
from torch.utils.data import Dataset
import json
from sklearn.preprocessing import normalize

# dataloader
class cocoData(Dataset):
    def __init__(self, annotations_file, class_name, transform=None):
        with open(annotations_file, 'r') as f:
            json_data = json.load(f)
        self.img_anns = json_data['annotations']
        self.class_name = class_name
        self.transform = transform

    def __len__(self):
        return len(self.img_anns)

    def __getitem__(self, idx):
        raw = []
        i = 0
        for keys in self.img_anns[idx]['keypoints']:
            if i == 2:
                i = 0
                continue
            elif i == 0:
                width = keys/1920
                raw.append(width)
                i += 1   
            elif i == 1:         
                height = keys/1080
                raw.append(height)   
                i += 1
        
        # norm = [float(i)/max(raw) for i in raw]
        # norm = [float(i)/sum(raw) for i in raw]
        
        # norm = normalize([raw])
        keypoint = torch.as_tensor(raw).float().squeeze(0)

        label = int(self.class_name[self.img_anns[idx]['category_id']])
        # print(keypoint.shape)
        if self.transform:
            keypoint = self.transform(keypoint)
            
        return keypoint, label
    
