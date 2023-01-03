import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from dataLoader import cocoData
from Net import Net
from Trainer import train
from Tester import *

# train_keypoints = 'data/etri/unit_add_none_train_key.json'
# valid_keypoints = 'data/etri/unit_add_none_test_key.json'
test_keypoints = './pose/output/pose_output.json'
# img_dir = './data/hyodol/threshold/video_to_images'
# # classes = { 3:'medicine', 31:'remote', 53:'fall'}
classes = {0:0, 3:1, 31:2, 53:3}
reverse_dict= dict(map(reversed,classes.items()))
pose_label = {0:'none', 1:'medicine', 2:'remote', 3:'fall_down'}
epochs = 50
batch_size = 64

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} ...')

# transform = transforms.Compose([
# 	# transforms.ToTensor(),
#     	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

# train_data = cocoData(train_keypoints, classes)
# valid_data = cocoData(valid_keypoints, classes)
test_data = cocoData(test_keypoints, classes)

# for i, (key, lab) in enumerate(test_data):
#     print(key.dtype, key.type(), type(key))
#     print(lab)
#     if i == 500:
#         break

# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,)
# valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True,)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True,)

net = Net(classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0001)

path = "./data/classification.pth" 
# train(device, criterion, optimizer, net, train_loader, valid_loader, epochs, path)
# test(device, net, test_loader, path)
test_species(device, net, test_loader, path, classes, pose_label)
