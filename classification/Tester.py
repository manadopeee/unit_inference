import torch
import time
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to test the model 
def test(device, model, test_loader, path): 
    # Load the model that we saved at the end of the training loop 
    model.load_state_dict(torch.load(path, map_location=device)) 
    
    running_accuracy = 0 
    total = 0 
    infer_time = 0.0
    start_time = time.time()

    with torch.no_grad(): 
        for data in test_loader: 
            inputs, outputs = data 
            inputs, outputs = inputs.to(device), outputs.to(device)
            
            if device == 'cpu':
                predicted_outputs = model(inputs) 
            else:
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                predicted_outputs = model(inputs) 
                ender.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                infer_time += (starter.elapsed_time(ender)/1000) / len(inputs)
                # print('Samples/Sec: ', (start.elapsed_time(end)/1000)/len(inputs))
            
            _, predicted = torch.max(predicted_outputs, 1) 
            total += outputs.size(0) 
            running_accuracy += (predicted == outputs).sum().item() 

        # print('Accuracy of the model based on the test set of', test_split ,'inputs is: %d %%' % (100 * running_accuracy / total))
        print('Accuracy of the model based on the test set of inputs is: %.2f %%' % (100 * running_accuracy / total))
        if device == 'cpu':
            print("Time: {:.4f}sec".format((time.time() - start_time)))
        else:
            print("'Final Throughput:': {:.5}sec".format(infer_time / len(test_loader)))
        

# Optional: Function to test which species were easier to predict  
def test_species(device, model, test_loader, path, classes, pose_label): 
    # Load the model that we saved at the end of the training loop 
    model.load_state_dict(torch.load(path, map_location=device)) 
    
    labels_length = len(classes) # how many labels of Irises we have. = 3 in our database. 
    labels_correct = list(0. for i in range(labels_length)) # list to calculate correct labels [how many correct setosa, how many correct versicolor, how many correct virginica] 
    labels_total = list(0. for i in range(labels_length))   # list to keep the total # of labels per type [total setosa, total versicolor, total virginica] 

    correct = 0
    total_len = 0
    y_true = []
    y_pred = []

    with torch.no_grad(): 
        for data in test_loader: 
            inputs, outputs = data 
            inputs, outputs = inputs.to(device), outputs.to(device)
            predicted_outputs = model(inputs) 
            _, predicted = torch.max(predicted_outputs, 1)
            y_true.extend(outputs.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            total_len += outputs.size(0)
            correct += (predicted == outputs).sum().item()

            label_correct_running = (predicted == outputs).squeeze() 
            
            for output, prediction in zip(outputs, predicted):
                if output == prediction:
                    labels_correct[output] += 1 
                labels_total[output] += 1  
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    class_names = ('none', 'medicine', 'remote', 'fall_down')
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
    # print(dataframe)
    plt.figure(figsize=(15, 8))

    # Create heatmap
    heatmap = sns.heatmap(dataframe, annot=True, cbar=True, cmap="YlGnBu", fmt="d")

    plt.title("Confusion Matrix")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel("True Class"), 
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    # plt.show()
    if not os.path.exists("./classification/output"):
        os.makedirs("./classification/output")
    plt.savefig('./classification/output/Confusion Matrix.png')
    
    # print(f'Accuracy of the network on the test images: {100 * correct // total_len} %')
    print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total_len))
    
    # label_list = list(classes.keys())
    for i in range(labels_length): 
        print('Accuracy to predict %10s : %5d / %5d = %.2f %%' % (pose_label[i], labels_correct[i], labels_total[i], 100 * labels_correct[i] / labels_total[i])) 
