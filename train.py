import argparse
import os
import shutil
import configparser
from datetime import datetime
import time

import torch
import torch.nn
from torch.utils.data import DataLoader
# import torch.utils.tensorboard
# from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from lib.data_preparation import read_and_generate_dataset
from lib.datasets import DatasetPEMS
from model.model_config import get_backbones
from lib.utils import compute_val_loss, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='configurations/PEMS04.conf',
                    help="configuration file path", required=False)
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
args = parser.parse_args()

# log dir
if os.path.exists('logs'):
    shutil.rmtree('logs')
    print('Remove log dir')

# read configuration
config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])

model_name = training_config['model_name']
ctx = training_config['ctx']
optimizer = training_config['optimizer']
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
merge = bool(int(training_config['merge']))

# select devices
if ctx.startswith('cpu'):
    ctx = torch.device("cpu")
elif ctx.startswith('gpu'):
    ctx = torch.device("cuda:" + ctx.split('-')[-1])
else:
    raise SystemError("error device input")

device = ctx

# import model
print('Model is %s' % (model_name))
if model_name == 'ASTGCN':
    from model.astgcn import ASTGCN as model
else:
    raise SystemExit('Wrong type of model!')

# make model params dir
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if 'params_dir' in training_config and training_config['params_dir'] != "None":
    params_path = os.path.join(training_config['params_dir'], model_name, timestamp)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)

# check parameters file
if os.path.exists(params_path) and not args.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s' % (params_path))

if __name__ == '__main__':
    start_time = time.perf_counter()

    # read all data from graph signal matrix file
    print("Reading data...")
    dataload_start_time = time.perf_counter()
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)
    dataload_end_time = time.perf_counter()
    print(f'Running time for data loading is {dataload_end_time - dataload_start_time:.2f} seconds')

    # test set ground truth
    true_value = (all_data['test']['target'].transpose((0, 2, 1))
                  .reshape(all_data['test']['target'].shape[0], -1))

    # training set data loader
    train_dataset = DatasetPEMS(all_data['train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # validation set data loader
    val_dataset = DatasetPEMS(all_data['val'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # why shuffle is False?

    # testing set data loader
    test_dataset = DatasetPEMS(all_data['test'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # save Z-score mean and std
    # stats_data = {}
    # for type_ in ['week', 'day', 'recent']:
    #     stats = all_data['stats'][type_]
    #     stats_data[type_ + '_mean'] = stats['mean']
    #     stats_data[type_ + '_std'] = stats['std']
    #
    # np.savez_compressed(
    #     os.path.join(params_path, 'stats_data'),
    #     **stats_data
    # )

    loss_function = torch.nn.MSELoss()

    all_backbones = get_backbones(args.config, adj_filename, device)
    # print(all_backbones[0][0]['cheb_polynomials'])

    num_of_features = 3
    num_of_timesteps = [[points_per_hour * num_of_weeks, points_per_hour],
                        [points_per_hour * num_of_days, points_per_hour],
                        [points_per_hour * num_of_hours, points_per_hour]]
    net = model(num_for_predict, all_backbones, num_of_vertices, num_of_features, num_of_timesteps, device)

    net = net.to(device)
    # it is the same as net.to(device)
    # i.e., to() for module is in place, which is different from tensor

    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # for params in net.parameters():
    #     torch.nn.init.normal_(params, mean=0, std=0.01)

    total_params = sum(p.numel() for p in net.parameters())
    train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total number of parameters is: %d" % total_params)
    print("Total number of trainable parameters is: %d" % train_params)

    group_num = 20
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        epoch_start_time = time.perf_counter()
        batch_start_time = epoch_start_time
        for i, [train_w, train_d, train_r, train_t] in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            train_w = train_w.to(device)
            train_d = train_d.to(device)
            train_r = train_r.to(device)
            train_t = train_t.to(device)

            outputs = net([train_w, train_d, train_r])

            loss = loss_function(outputs, train_t)  # loss is a tensor on the same device as outpus and train_t
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # type of running_loss is float, loss.item() is a float on CPU

            if i % group_num == group_num - 1:
                batch_end_time = time.perf_counter()
                print(f'[{epoch:d}, {i + 1:5d}] loss: {running_loss / group_num:.2f}, \
                        time: {batch_end_time - batch_start_time:.2f}')
                running_loss = 0.0
                batch_start_time = batch_end_time

        epoch_end_time = time.perf_counter()
        print(f'Epoch cost {epoch_end_time - epoch_start_time:.2f} seconds')

        # probably not need to run this after every epoch
        with torch.no_grad():
            # compute validation loss
            compute_val_loss(net, val_loader, loss_function, None, epoch, device)

        # testing
            evaluate(net, test_loader, true_value, num_of_vertices, None, epoch, device)

    end_time = time.perf_counter()
    print(f'Total running time is {end_time - start_time:.2f} seconds.')
