import time

import pandas as pd
import psutil
from scipy.stats import spearmanr

import pickle

from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset

import json

from basic_model_mm import mmKcatPrediction
from pretrain_trfm import *
from early_stop import *


mask_array = np.array([[True, True, True, True],
                       [True, True, True, False],
                       [True, True, False, True],
                       [True, True, False, False],  # Normal
                       [False, True, True, True],
                       [False, True, True, False],
                       [False, True, False, True],
                       [False, True, False, False],
                       [True, False, True, True],
                       [True, False, True, False],
                       [True, False, False, True],
                       [True, False, False, False],
                       [False, False, True, True],
                       [False, False, True, False],
                       [False, False, False, True],
                       [False, False, False, False]])  # Abnormal


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

dataset = CustomizedDataset('../data/concat_test_dataset_final_latest.json',
                            '../data/concat_test_graph_x_latest.pkl',
                            '../data/concat_test_graph_edge_index_latest.pkl')

device = torch.device('cuda:2')
is_ground_truth_recorded = False
print('basic_model_mm')

test_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False,
                             collate_fn=costomized_collate_fn)

print(f'{len(dataset)} test data in total')

model = mmKcatPrediction(device=device, batch_size=32, mode='test',
                         is_enzymes_graph_masked=False, is_products_masked=True).to(device)
# model.load_state_dict(torch.load('../ckpt/concat_71.05077004432678_checkpoint.pth'))
total_params = sum(p.numel() for p in model.parameters()) / (10 ** 6)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / (10 ** 6)
print(total_params, trainable_params, total_params - trainable_params)
model.eval()

df = pd.DataFrame(columns=['Mask', 'MAE', 'RMSE', 'R2', 'SRCC', 'p-value'])
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=300, eta_min=0)
# early_stop = EarlyStopping(patience=5, verbose=True, dataset_name='concat')
# writer = SummaryWriter('../log/experiment1')
# epoch = 300
# training_iteration_count = 0
# validation_iteration_count = 0
start_time = time.time()
start_mem = psutil.Process().memory_info().rss

with torch.no_grad():

    for mask in mask_array:

        print(f'Testing {mask}....')

        SAE = 0.0
        ground_truths = []
        predicted_values = []

        # Set mask
        model.test_mask = mask

        for batch_idx, data in enumerate(test_loader):
            (loss, loss_x1, loss_x2, loss_x3, loss_x4, loss_x5,
             predicted_x1, predicted_x2, predicted_x3, predicted_x4, predicted_x5) = model(data)
            predicted_list = [predicted_kcat[0] for predicted_kcat in predicted_x5.tolist()]
            kcat_list = [kcat.item() for kcat in data[4]]
            SAE += sum(np.abs(np.array(predicted_list) - np.array(kcat_list)))
            ground_truths += (kcat_list)
            predicted_values += (predicted_list)

        MAE = SAE / len(dataset)
        RMSE = np.sqrt(mean_squared_error(ground_truths, predicted_values))
        R2 = r2_score(ground_truths, predicted_values)
        correlation, p_value = spearmanr(ground_truths, predicted_values)
        print(f'Mask: {mask}\n'
              f'MAE: {MAE}\n'
              f'RMSE: {RMSE}\n'
              f'R2: {R2}\n'
              f'SRCC: {correlation}\n'
              f'p-value: {p_value}')

        new_data = {'Mask': mask, 'MAE': MAE, 'RMSE': RMSE, 'R2': R2, 'SRCC': correlation, 'p-value': p_value}
        df = df.append(new_data, ignore_index=True)

        # if not is_ground_truth_recorded:
        #     with open('./test_records/latest/with_aux/ground_truth.pkl', 'wb') as file:
        #         pickle.dump(ground_truths, file)
        #     is_ground_truth_recorded = True
        #
        # with open('test_records/latest/with_aux/' + str(mask[0]) + str(mask[1]) + str(mask[2]) + str(mask[3]) +
        #           '.pkl', 'wb') as file:
        #     pickle.dump(predicted_values, file)

    # df.to_csv('./test_records/2.csv')

end_time = time.time()
end_mem = psutil.Process().memory_info().rss

elapsed_time = end_time - start_time
additional_mem = (end_mem - start_mem) / (1024 * 1024)
print(elapsed_time, additional_mem)
