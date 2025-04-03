import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import copy
import logging

# Code for Activations and Labels

def get_data(n, d, label_fun, boolean=False):
    if boolean:
        data = (torch.randint(0, 2, size=(n,d))*2 - 1).float()
    else:
        data = torch.normal(0,1, size=(n,d))
    y = label_fun(data)
    return (data, y)

class Net(nn.Module):
    def __init__(self, d, m, activation):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.fc2 = nn.Linear(m, 1)
        self.activation = activation
        self.m = m
        # self.k = k

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def path_norm(self):
        return torch.matmul(torch.square(self.fc2.weight), torch.sum(torch.square(self.fc1.weight), dim=1))

def corr_loss(y_true, y_pred):
    return -torch.mean(y_true * y_pred)

def train(net, all_data, hps, problem_params, opts, optimizer, sn, ferr, scaler, lang=0, lpath=0):
    inputs, labels, test_inputs, test_labels = all_data["inputs"], all_data['labels'], all_data['test_inputs'], all_data['test_labels']
    label_fun, boolean, corr = problem_params["label_fun"], problem_params["boolean"], problem_params["corr"]
    batch_size, epoch, fresh_train_data, fresh_test_data, seed = hps["batch"], hps["epoch"], hps["fresh_train"], hps["fresh_test"], hps["seed"]
    verbose, k_indices = opts["print"], opts["k_indices"]
    n = len(labels)
    d = len(inputs.T)
    losses = []
    train_err = []
    test_err = []
    # test_module(-1, net, 0, d, label_fun, test_inputs, test_labels, fresh_data=fresh_test_data, verbose=True)
    for t in range(epoch):
        # If we are not reusing data, get fresh data.
        if fresh_train_data:
            torch.manual_seed(2*t + seed) # This is to ensure consistency across runs of diff widths.
            inputs, labels = get_data(n, d, label_fun, boolean=boolean)

        # Save the first k_indices coordinates of all neurons for later inspection and analysis.
        # Run an epoch of SGD
        for i in range(np.ceil(n/batch_size).astype(int)):
            #sn.append(np.copy(net.fc1.weight.detach().numpy()[:, :k_indices]))
            #S = range(batch_size*i, batch_size*(i + 1))
            np.random.seed(seed + t + i)
            S = np.random.choice(n,batch_size, replace=False)
            batch_inputs = inputs[S,:]
            batch_labels = labels[S].reshape(batch_size,1)
            with autocast():
                batch_outputs = net(batch_inputs)
                if corr:
                    loss = corr_loss(batch_outputs, batch_labels.reshape(batch_size,1))
                else:
                    loss = torch.nn.MSELoss()(batch_outputs, batch_labels.reshape(batch_size,1))
                if lpath > 0:
                    loss += lpath*net.path_norm()
            losses.append(loss.detach().numpy())

            optimizer.zero_grad()
            scaler.scale(loss).backward()      # Scales loss to avoid underflow
            scaler.step(optimizer)             # Unscales and applies optimizer step
            scaler.update()                    # Updates scale for next step
            # loss.backward()
            # optimizer.step()

            # Normalize weights to stay on the sphere
            with torch.no_grad():
                if lang > 0:
                    net.fc1.weight += hps["lr"]*torch.normal(0, lang, size=(M, d))
                if lpath == 0:
                    norms = torch.norm(net.fc1.weight, dim=1, keepdim=True)
                    net.fc1.weight.copy_(net.fc1.weight / norms)
        # if t % 2 == 0:
        sn.append(np.copy(net.fc1.weight.detach().numpy()[:, :k_indices + 1]))
        # Every 10th step, print training stats
        if t % opts["test_freq"] == 0:
            ferr_out, test_error = test_module(t*int(n/batch_size) + i, net, loss, d, label_fun, test_inputs, test_labels, verbose=verbose)
            # train_error = test_module(t*int(n/batch_size) + i, net, loss, d, label_fun, train_inputs, train_labels, verbose=verbose)
            # train_err.append(train_error)
            ferr.append(ferr_out)
            test_err.append(test_error)
            train_err.append(np.mean(losses[-int(n/batch_size):]))
    

        if t == epoch - 1:
            test_module(t, net, loss, d, label_fun, test_inputs, test_labels, verbose=True)
    return losses, test_err, train_err # Note: losses has length #SGD steps, test_err has length #epochs/10

def test_module(epoch, net, train_error, d, label_fun, test_inputs, test_labels, fresh_data=False, seed=2358, boolean=False, verbose=True):
    if fresh_data:
      torch.manual_seed(epoch + 5 + seed)
      n_test = 1000
      test_inputs, test_labels = get_data(n_test, d, label_fun, boolean=boolean)

    n_test = len(test_labels)
    with autocast():
        outputs = net(test_inputs)
        test_error = torch.nn.MSELoss()(outputs, test_labels.reshape(n_test,1))

    dictionary = dict()
    dictionary['epoch'] = epoch
    dictionary['t-err'] = test_error
    dictionary['tr-err'] = train_error
    if verbose:
        printformat(dictionary)
    return outputs.detach().numpy(), test_error.detach().numpy()

def printformat(dictionary):
    for key in dictionary.keys():
        print(key, end=': ')
        if key == 'epoch':
            print(format(dictionary[key]), end=' | ')
        else:
            print(format(dictionary[key], ".3f"), end=' | ')
    print('\n')


# Standard MF initialization
# If signed, then 2nd layers weights will have random signs
def normal_init(net, m, d, signed=False, scale=1, backup=None):
    W1 = scale*torch.normal(0,1/np.sqrt(d), size=(m,d))
    if backup is not None:
        W1= backup[:m]
    net.fc1.weight = torch.nn.Parameter(W1)
    net.fc1.bias = torch.nn.Parameter(torch.zeros(m))
    if signed:
       pmones = torch.tensor([1, -1]).repeat(1, int(m/2))
       net.fc2.weight = torch.nn.Parameter((1.0/m)*pmones)
    else:
       net.fc2.weight = torch.nn.Parameter((1.0/m)*torch.ones(1, int(m)))
    net.fc2.bias = torch.nn.Parameter(torch.zeros(1))
    return copy.deepcopy(net)

# Biased initialization in e0 or e1 direction (0 - indexed)
def weird_init(net, m, d, signed=False, scale=1, backup=None):
    W1 = scale*torch.normal(0,1/np.sqrt(d), size=(m,d))
    if backup is not None:
        m2 = len(backup)
        W1[:m2]= backup
    W1[:, 0] = torch.tensor([1, 2]).repeat(1, int(m/2)) # W1[:, 1] = 0.5*torch.ones(m)
    norms = torch.norm(W1, dim=1, keepdim=True)
    W1=W1/norms
    net.fc1.weight = torch.nn.Parameter(W1)
    net.fc1.bias = torch.nn.Parameter(torch.zeros(m))
    if signed:
       pmones = torch.tensor([1, -1]).repeat(1, int(m/2))
       net.fc2.weight = torch.nn.Parameter((1.0/m)*pmones)
    else:
       net.fc2.weight = torch.nn.Parameter((1.0/m)*torch.ones(1, int(m)))
    net.fc2.bias = torch.nn.Parameter(torch.zeros(1))
    return copy.deepcopy(net)

def init_and_train(width, backup, all_data, hps, problem_params, opts, lang=0, lpath=0):
    new_net = Net(problem_params["d"], width, problem_params["activation"]).to(device)
    normal_init(new_net, width, problem_params["d"], backup=copy.deepcopy(backup), signed=problem_params["signed"])
    # Train
    p = list(new_net.parameters())
    optimizer = optim.SGD([p[0]], lr=hps["lr"]*width) # Just train first layer, LR gets multiplied by m because MF velocity scales like m * derivative of loss.
    if lpath > 0:
        optimizer = optim.SGD([p[0], p[2]], lr=hps["lr"]*width)
        logging.log("Training with lpath = %s" % lpath)
    scaler = GradScaler()  # For scaling gradients safely in float16
    epochs = hps["epoch"]
    div = 10
    L = []
    R = []
    Lavg = []
    for i in range(div):
        hps["epoch"] = int(epochs/div)
        if i == div - 1:
            hps["epoch"] = epochs - (div - 1)*int(epochs/div)
        saved_neurons = []
        ferrouts = []
        (Li, Ri, Lavgi) = train(new_net, all_data, hps, problem_params, opts, optimizer, saved_neurons, ferrouts, scaler, lang=lang, lpath=lpath)
        np.save("results/%s/data/saved_dynamics_d_%s_m_%s_%s_%s" % (problem_params["alias"], problem_params["d"], width, problem_params["name"], i), np.array(saved_neurons))
        np.save("results/%s/data/xout_d_%s_m_%s_%s_%s" % (problem_params["alias"], problem_params["d"], width, problem_params["name"], i), np.array(ferrouts))
        L = L + Li
        R = R + Ri
        Lavg = Lavg + Lavgi
    hps["epoch"] = epochs
    return saved_neurons, L, R, Lavg