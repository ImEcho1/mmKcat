import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import json

from basic_model_mm import mmKcatPrediction
from pretrain_trfm import *
from early_stop import *


class CustomizedDataset(Dataset):
    def __init__(self, seq_path, graph_x_path, graph_edge_index_path):
        with open(seq_path, 'r') as file:
            data_dict = json.load(file)
        file.close()

        # Prepare sequence-relevant data
        print('Preparing sequence data...')
        chosen_seq_data = {k: v for k, v in data_dict.items() if
                           len(v['Substrate_Smiles']) != 0 and v['Sequence_Rep'] is not None}
        self.seq_data = list(chosen_seq_data.values())
        assert len(data_dict) == len(self.seq_data)

        # Prepare graph-relevant data
        print('Preparing graph data...')
        chosen_graph_data = []
        with open(graph_x_path, 'rb') as file:
            graph_x = pickle.load(file)
        file.close()
        with open(graph_edge_index_path, 'rb') as file:
            graph_edge_index = pickle.load(file)
        file.close()
        for index, cur_graph in enumerate(zip(graph_x, graph_edge_index)):
            chosen_graph_data.append(cur_graph)
        self.graph_data = chosen_graph_data

        assert len(self.seq_data) == len(self.graph_data)

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        seq_item, graph_item = self.seq_data[idx], self.graph_data[idx]
        return seq_item, graph_item


def costomized_collate_fn(batch):
    substrates_list = []
    enzymes_rep_list = []
    enzymes_graph_list = []
    products_list = []
    kcats_list = []
    seq_data = []
    graph_data = []
    for data_item in batch:
        seq_data.append(data_item[0])
        graph_data.append(data_item[1])
    for _data in seq_data:
        substrates_list.append(_data['Substrate_Smiles'])
        enzymes_rep_list.append(torch.tensor(_data['Sequence_Rep']).unsqueeze(dim=0))
        products_list.append(_data['Product_Smiles'])
        kcats_list.append(torch.tensor(float(_data['Value'])).unsqueeze(-1))
    for _data in graph_data:
        enzymes_graph_list.append(_data)
    return [substrates_list, enzymes_rep_list, enzymes_graph_list, products_list, kcats_list]


# Set random seed
torch.manual_seed(131)
np.random.seed(131)

dataset = CustomizedDataset('../data/concat_train_dataset_final_latest.json',
                            '../data/concat_train_graph_x_latest.pkl',
                            '../data/concat_train_graph_edge_index_latest.pkl')
train_ratio = 0.9
train_size = int(train_ratio * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
device = torch.device('cuda:3')
print('basic_model_mm')

training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False,
                             collate_fn=costomized_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True, drop_last=False, collate_fn=costomized_collate_fn)

model = mmKcatPrediction(device=device, batch_size=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=300, eta_min=0)
early_stop = EarlyStopping(patience=5, verbose=True, dataset_name='concat')
writer = SummaryWriter('../log/experiment1')
epoch = 300
training_iteration_count = 0
validation_iteration_count = 0

for e in range(0, epoch):

    # Train model
    model.train()
    model.sub_seq_channel.eval()
    model.prod_seq_channel.eval()

    train_loss_total = 0.0
    train_loss1_total = 0.0
    train_loss2_total = 0.0
    train_loss3_total = 0.0
    train_loss4_total = 0.0
    train_loss5_total = 0.0
    for batch_idx, data in enumerate(training_loader):
        (loss, loss_x1, loss_x2, loss_x3, loss_x4, loss_x5,
         predicted_x1, predicted_x2, predicted_x3, predicted_x4, predicted_x5) = model(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_total += loss.item()
        train_loss1_total += loss_x1.item()
        train_loss2_total += loss_x2.item()
        train_loss3_total += loss_x3.item()
        train_loss4_total += loss_x4.item()
        train_loss5_total += loss_x5.item()
        writer.add_scalar('train_loss_iteration', loss, training_iteration_count)
        writer.add_scalar('train_loss_x1_iteration', loss_x1, training_iteration_count)
        writer.add_scalar('train_loss_x2_iteration', loss_x2, training_iteration_count)
        writer.add_scalar('train_loss_x3_iteration', loss_x3, training_iteration_count)
        writer.add_scalar('train_loss_x4_iteration', loss_x4, training_iteration_count)
        writer.add_scalar('train_loss_x5_iteration', loss_x5, training_iteration_count)
        print(f'epoch: {e}, iteration: {training_iteration_count}\n'
              f'train_loss_iteration: {loss:.4f}, train_loss_x1_iteration: {loss_x1:.4f}, train_loss_x2_iteration: {loss_x2:.4f}\n'
              f'train_loss_x3_iteration: {loss_x3:.4f}, train_loss_x4_iteration: {loss_x4:.4f}, train_loss_x5_iteration: {loss_x5:.4f}')
        training_iteration_count += 1

    writer.add_scalar('train_loss_epoch', train_loss_total, e)
    writer.add_scalar('train_loss1_epoch', train_loss1_total, e)
    writer.add_scalar('train_loss2_epoch', train_loss2_total, e)
    writer.add_scalar('train_loss3_epoch', train_loss3_total, e)
    writer.add_scalar('train_loss4_epoch', train_loss4_total, e)
    writer.add_scalar('train_loss5_epoch', train_loss5_total, e)

    # Valid model
    model.eval()
    valid_loss_total = 0.0
    SAE_x1 = 0.0
    SAE_x2 = 0.0
    SAE_x3 = 0.0
    SAE_x4 = 0.0
    SAE_x5 = 0.0
    ground_truths = []
    predicted_values_x1 = []
    predicted_values_x2 = []
    predicted_values_x3 = []
    predicted_values_x4 = []
    predicted_values_x5 = []
    with (torch.no_grad()):
        for batch_idx, data in enumerate(valid_loader):
            (loss, loss_x1, loss_x2, loss_x3, loss_x4, loss_x5,
             predicted_x1, predicted_x2, predicted_x3, predicted_x4, predicted_x5) = model(data)
            predicted_x1_list = [predicted_kcat[0] for predicted_kcat in predicted_x1.tolist()]
            predicted_x2_list = [predicted_kcat[0] for predicted_kcat in predicted_x2.tolist()]
            predicted_x3_list = [predicted_kcat[0] for predicted_kcat in predicted_x3.tolist()]
            predicted_x4_list = [predicted_kcat[0] for predicted_kcat in predicted_x4.tolist()]
            predicted_x5_list = [predicted_kcat[0] for predicted_kcat in predicted_x5.tolist()]
            kcat_list = [kcat.item() for kcat in data[4]]
            SAE_x1 += sum(np.abs(np.array(predicted_x1_list) - np.array(kcat_list)))
            SAE_x2 += sum(np.abs(np.array(predicted_x2_list) - np.array(kcat_list)))
            SAE_x3 += sum(np.abs(np.array(predicted_x3_list) - np.array(kcat_list)))
            SAE_x4 += sum(np.abs(np.array(predicted_x4_list) - np.array(kcat_list)))
            SAE_x5 += sum(np.abs(np.array(predicted_x5_list) - np.array(kcat_list)))
            ground_truths += (kcat_list)
            predicted_values_x1 += (predicted_x1_list)
            predicted_values_x2 += (predicted_x2_list)
            predicted_values_x3 += (predicted_x3_list)
            predicted_values_x4 += (predicted_x4_list)
            predicted_values_x5 += (predicted_x5_list)
            valid_loss_total += loss.item()
            writer.add_scalar('validation_loss_iteration', loss, validation_iteration_count)
            writer.add_scalar('validation_loss_x1_iteration', loss_x1, validation_iteration_count)
            writer.add_scalar('validation_loss_x2_iteration', loss_x2, validation_iteration_count)
            writer.add_scalar('validation_loss_x3_iteration', loss_x3, validation_iteration_count)
            writer.add_scalar('validation_loss_x4_iteration', loss_x4, validation_iteration_count)
            writer.add_scalar('validation_loss_x5_iteration', loss_x5, validation_iteration_count)
            validation_iteration_count += 1
        # ground_truths = np.array(ground_truths)
        # predicted_values_x1 = np.array(predicted_values_x1)
        # predicted_values_x2 = np.array(predicted_values_x2)
        assert len(ground_truths) == len(predicted_values_x1) == len(predicted_values_x2) \
        == len(predicted_values_x3) == len(predicted_values_x4) == len(predicted_values_x5)
        MAE_x1 = SAE_x1 / len(valid_dataset)
        MAE_x2 = SAE_x2 / len(valid_dataset)
        MAE_x3 = SAE_x3 / len(valid_dataset)
        MAE_x4 = SAE_x4 / len(valid_dataset)
        MAE_x5 = SAE_x5 / len(valid_dataset)
        rmse_x1 = np.sqrt(mean_squared_error(ground_truths, predicted_values_x1))
        rmse_x2 = np.sqrt(mean_squared_error(ground_truths, predicted_values_x2))
        rmse_x3 = np.sqrt(mean_squared_error(ground_truths, predicted_values_x3))
        rmse_x4 = np.sqrt(mean_squared_error(ground_truths, predicted_values_x4))
        rmse_x5 = np.sqrt(mean_squared_error(ground_truths, predicted_values_x5))
        r2_x1 = r2_score(ground_truths, predicted_values_x1)
        r2_x2 = r2_score(ground_truths, predicted_values_x2)
        r2_x3 = r2_score(ground_truths, predicted_values_x3)
        r2_x4 = r2_score(ground_truths, predicted_values_x4)
        r2_x5 = r2_score(ground_truths, predicted_values_x5)
        print(f'epoch: {e}, validation_loss_epoch: {valid_loss_total:.4f}\n'
              f'MAE_x1: {MAE_x1:.4f}, MAE_x2: {MAE_x2:.4f}, MAE_x3: {MAE_x3:.4f}, MAE_x4: {MAE_x4:.4f} MAE_x5: {MAE_x5:.4f}\n'
              f'rmse_x1: {rmse_x1:.4f}, rmse_x2: {rmse_x2:.4f}, rmse_x3: {rmse_x3:.4f}, rmse_x4: {rmse_x4:.4f}, rmse_x5: {rmse_x5:.4f}\n'
              f'r2_x1: {r2_x1:.4f}, r2_x2: {r2_x2:.4f}, r2_x3: {r2_x3:.4f}, r2_x4: {r2_x4:.4f}, r2_x5: {r2_x5:.4f}')
        print('-----------------------------------------------------------------------------------------------------')
        writer.add_scalar('validation_loss_epoch', valid_loss_total, e)
        writer.add_scalar('SAE_x1', SAE_x1, e)
        writer.add_scalar('SAE_x2', SAE_x2, e)
        writer.add_scalar('SAE_x3', SAE_x3, e)
        writer.add_scalar('SAE_x4', SAE_x4, e)
        writer.add_scalar('SAE_x5', SAE_x5, e)
        writer.add_scalar('MAE_x1', MAE_x1, e)
        writer.add_scalar('MAE_x2', MAE_x2, e)
        writer.add_scalar('MAE_x3', MAE_x3, e)
        writer.add_scalar('MAE_x4', MAE_x4, e)
        writer.add_scalar('MAE_x5', MAE_x5, e)
        writer.add_scalar('rmse_x1', rmse_x1, e)
        writer.add_scalar('rmse_x2', rmse_x2, e)
        writer.add_scalar('rmse_x3', rmse_x3, e)
        writer.add_scalar('rmse_x4', rmse_x4, e)
        writer.add_scalar('rmse_x5', rmse_x5, e)
        writer.add_scalar('r2_x1', r2_x1, e)
        writer.add_scalar('r2_x2', r2_x2, e)
        writer.add_scalar('r2_x3', r2_x3, e)
        writer.add_scalar('r2_x4', r2_x4, e)
        writer.add_scalar('r2_x5', r2_x5, e)

        # Decide whether to early stop
        early_stop(val_loss=valid_loss_total, model=model, path='../ckpt/cuda3_2/')

# Save final model
torch.save(model.state_dict(), '../ckpt/cuda3_2/final_model.pt')
print('Training finished.')
