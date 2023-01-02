import torch
from tqdm import tqdm

# Training Function 
def train(device, loss_fn, optimizer, model, train_loader, validate_loader, num_epochs, path): 
    best_accuracy = 0.0 
     
    print("Begin training...") 
    for epoch in range(1, num_epochs+1): 
        running_train_loss = 0.0 
        running_accuracy = 0.0 
        running_vall_loss = 0.0 
        total = 0 
 
        # Training Loop 
        for data in tqdm(train_loader): 
        #for data in enumerate(train_loader, 0): 
            inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs] 
            inputs, outputs = inputs.to(device), outputs.to(device)
            optimizer.zero_grad()   # zero the parameter gradients          
            predicted_outputs = model(inputs)   # predict output from the model 
            train_loss = loss_fn(predicted_outputs, outputs)   # calculate loss for the predicted output  
            train_loss.backward()   # backpropagate the loss 
            optimizer.step()        # adjust parameters based on the calculated gradients 
            running_train_loss +=train_loss.item()  # track the loss value 
 
        # Calculate training loss value 
        train_loss_value = running_train_loss/len(train_loader) 
 
        # Validation Loop 
        with torch.no_grad(): 
            model.eval() 
            for data in validate_loader: 
               inputs, outputs = data 
               inputs, outputs = inputs.to(device), outputs.to(device)
               predicted_outputs = model(inputs) 
               val_loss = loss_fn(predicted_outputs, outputs) 
             
               # The label with the highest value will be our prediction 
               _, predicted = torch.max(predicted_outputs, 1) 
               running_vall_loss += val_loss.item()  
               total += outputs.size(0) 
               running_accuracy += (predicted == outputs).sum().item() 
 
        # Calculate validation loss value 
        val_loss_value = running_vall_loss/len(validate_loader) 
                
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
        accuracy = (100 * running_accuracy / total)     
 
        # Save the model if the accuracy is the best 
        if accuracy > best_accuracy: 
            saveModel(path, model) 
            best_accuracy = accuracy 
         
        # Print the statistics of the epoch 
        print('Completed training batch', epoch, 
            '/ Training Loss is: %.4f' %train_loss_value, 
            '/ Validation Loss is: %.4f' %val_loss_value, 
            '/ Accuracy is %d %%' % (accuracy))
    print('Finished Training\n') 
    
# Function to save the model 
def saveModel(path, model): 
    torch.save(model.state_dict(), path) 
