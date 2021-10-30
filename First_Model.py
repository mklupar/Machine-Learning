import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import math

if __name__ == '__main__':
    '''Download and explore data'''

    dataframe = pd.read_csv('insurance.csv')
    num_rows, num_cols = dataframe.shape  # finds total number of rows ans columns

    datatypeDict = dict(dataframe.dtypes)  # dict of column titles and data types

    catergorical_cols = ['sex', 'smoker', 'region']

    title = list(datatypeDict)
    output_cols = title[6:]  # output columns
    input_cols = title[:-1]  # target columns

    '''Prepare the dataset for training'''


    def dataframe_to_arrays(dataframe):  # Converts pandas dataframe to an array
        # make a copy of the original dataframe
        dataframe1 = dataframe.copy(deep=True)
        # convert non-numeric categorical columns to numbers
        for col in catergorical_cols:
            dataframe1[col] = dataframe1[col].astype('category').cat.codes
        # Extract input and outputs as numpy array
        inputs_array = dataframe1[input_cols].to_numpy()
        targets_array = dataframe1[output_cols].to_numpy()
        return inputs_array, targets_array


    input_array, target_array = dataframe_to_arrays(dataframe)

    # Convert a float64 array to a float32 array to improve computation time
    input_array = np.float32(input_array)
    target_array = np.float32(target_array)

    inputs = torch.from_numpy(input_array)
    targets = torch.from_numpy(target_array)

    dataset = TensorDataset(inputs, targets)  # allows us to start the dataloader

    # Split data into training data and validation data
    val_percent = 0.2
    val_size = int(num_rows * val_percent)
    train_size = num_rows - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 25
    ### Might be an error around here.
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)

    in_features = len(input_cols)
    out_features = len(output_cols)


    class InsuranceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)  # creates weights and bias

        def forward(self, xb):
            out = self.linear(xb)
            return out

        def training_step(self, batch):
            inputs_, targets_ = batch
            out = self(inputs_)
            loss = F.mse_loss(out, targets_)
            return loss

        def validation_step(self, batch):
            inputs_, targets_ = batch
            out = self(inputs_)
            loss = F.mse_loss(out, targets_)
            return {'val_loss': loss.detach()}

        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
            return {'val_loss': epoch_loss.item()}

        def epoch_end(self, epoch, result, num_epochs):
            # Print result every 20th epoch
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                print("Epoch [{}], val_loss: {:.4f}".format(epoch + 1, result['val_loss']))


    model = InsuranceModel()

    '''Train the Model!'''


    def evaluate(model_, val_loader_):
        outputs = [model_.validation_step(batch) for batch in val_loader_]
        return model_.validation_epoch_end(outputs)


    def fit(epochs_, lr, model_, train_loader_, val_loader_, opt_func=torch.optim.SGD):
        optimizer = opt_func(model_.parameters(), lr)
        history = []

        for epoch in range(epochs_):
            for batch in train_loader:
                loss = model_.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            results = evaluate(model_, val_loader_)
            model_.epoch_end(epoch, results, epochs)
            history.append(results)
        return history


    epochs = 1500
    lr = 1e-4
    # history1 = fit(epochs, lr, model, train_loader, val_loader)
    # torch.save(model.state_dict(), 'Insurance-Model.pth')
    model2 = InsuranceModel()
    model2.load_state_dict(torch.load('Insurance-Model.pth'))
    val_loss = evaluate(model2, val_loader)['val_loss']

    '''Make Predictions using model'''


    def predict_single(input_, target, model_):
        inputs_ = input_.unsqueeze(0)
        predictions = model2(inputs_)
        prediction = predictions[0].detach()
        print("Input:", input_)
        print("Target:", target)
        print("Prediction:", prediction)

    input1, target1 = val_ds[0]
    predict_single(input1, target1, model2)
