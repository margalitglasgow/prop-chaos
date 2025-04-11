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

def extend_data(old_data, d, boolean=False):
    old_d = old_data.shape[1]
    n = old_data.shape[0]
    if boolean: 
        new_data = (torch.randint(0, 2, size=(n,d))*2 - 1).float()
    else:
        new_data = torch.normal(0,1, size=(n,d))
    new_data[:, :old_d] = old_data
    return new_data

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
    accum_steps = hps["accum"]
    n = len(labels)
    d = len(inputs.T)
    losses = []
    train_err = []
    test_err = []
    #Initialize weights to be on the sphere.
    with torch.no_grad():
        norms = net.fc1.weight.norm(dim=1, keepdim=True)
        net.fc1.weight.copy_(net.fc1.weight / norms)
    # test_module(-1, net, 0, d, label_fun, test_inputs, test_labels, fresh_data=fresh_test_data, verbose=True)
    # logging.info("Starting div with %s epochs" % epoch)
    for t in range(epoch):
        # Logging
        ind = min(k_indices + 4, d)
        sn.append(np.copy(net.fc1.weight.detach().cpu().numpy()[:, :ind]))
        # logging.info("Logged sn at epoch %s" % t)
        if t % opts["test_freq"] == 0:
            net.eval()
            ferr_out, test_error = test_module(t*int(n/batch_size), net, 0, d, label_fun, test_inputs, test_labels, hps, verbose=verbose)
            ferr.append(ferr_out)
            test_err.append(test_error)
            train_err.append(np.mean(losses[-int(n/batch_size):]))
            net.train()

        # If we are not reusing data, get fresh data.
        if fresh_train_data:
            torch.manual_seed(2*t + seed) # This is to ensure consistency across runs of diff widths.
            inputs, labels = get_data(n, d, label_fun, boolean=boolean)

        # Run an epoch of SGD
        batch_steps = np.ceil(n/batch_size).astype(int)
        # logging.info("Number of bacth steps: %s, batch size %s" % (batch_steps, batch_size))
        for i in range(batch_steps):
            #sn.append(np.copy(net.fc1.weight.detach().numpy()[:, :k_indices]))
            S = range(batch_size*i, batch_size*(i + 1))
            # np.random.seed(seed + t + i)
            # S = np.random.choice(n,batch_size, replace=False)
            batch_inputs = inputs[S,:]
            batch_labels = labels[S].reshape(batch_size,1)
            with autocast():
                batch_outputs = net(batch_inputs)
                if corr:
                    loss = corr_loss(batch_outputs, batch_labels.reshape(batch_size,1))/accum_steps
                else:
                    loss = torch.nn.MSELoss()(batch_outputs, batch_labels.reshape(batch_size,1))/accum_steps
                if lpath > 0:
                    loss += lpath*net.path_norm()
                losses.append(loss.item()*accum_steps)

            scaler.scale(loss).backward()      # Scales loss to avoid underflow
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)             # Unscales and applies optimizer step
                scaler.update()                    # Updates scale for next step
                optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
                # Normalize weights to stay on the sphere
                with torch.no_grad():
                    if lang > 0:
                        torch.manual_seed(epoch + 42)
                        torch.cuda.manual_seed_all(epoch + 42)
                        rand_pert = torch.normal(0, lang, size=(1048576, d)).to(device)
                        net.fc1.weight += hps["lr"]*rand_pert[:M, :]
                    if lpath == 0:
                        norms = net.fc1.weight.norm(dim=1, keepdim=True)
                        net.fc1.weight.copy_(net.fc1.weight / norms)
                # logging.info(str(torch.cuda.memory_summary()))

    return losses, test_err, train_err # Note: losses has length #SGD steps, test_err has length #epochs/10

def batched_forward(model, inputs, batch_size=4096):
    outputs = []
    with torch.no_grad():
        for i in range(0, inputs.size(0), batch_size):
            batch = inputs[i:i+batch_size]
            with autocast():
                out = model(batch)
            outputs.append(out)

    return torch.cat(outputs, dim=0)


def test_module(epoch, net, train_error, d, label_fun, test_inputs, test_labels, hps, fresh_data=False, seed=2358, boolean=False, verbose=True):
    if fresh_data:
      torch.manual_seed(epoch + 5 + seed)
      n_test = 1000
      test_inputs, test_labels = get_data(n_test, d, label_fun, boolean=boolean)

    n_test = len(test_labels)

    accum_steps = hps["accum"]

    with autocast():
        outputs = batched_forward(net, test_inputs, batch_size=int(16384/accum_steps))
        test_error = torch.nn.MSELoss()(outputs, test_labels.reshape(n_test,1))

    dictionary = dict()
    dictionary['epoch'] = epoch
    dictionary['t-err'] = test_error
    # dictionary['tr-err'] = train_error
    if verbose:
        printformat(dictionary)
    return outputs.detach().cpu().numpy(), test_error.item()

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
    new_net = Net(problem_params["d"], width, problem_params["activation"])
    normal_init(new_net, width, problem_params["d"], backup=copy.deepcopy(backup), signed=problem_params["signed"])
    new_net = new_net.to(device)
    # Train
    p = list(new_net.parameters())
    optimizer = optim.SGD([p[0]], lr=hps["lr"]*width) # Just train first layer, LR gets multiplied by m because MF velocity scales like m * derivative of loss.
    # if lpath > 0:
    #     optimizer = optim.SGD([p[0], p[2]], lr=hps["lr"]*width)
    #     logging.info("Training with lpath = %s" % lpath)
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
        all_data["inputs"] = all_data["inputs"].to(device)
        all_data["labels"] = all_data["labels"].to(device)
        all_data["test_inputs"] = all_data["test_inputs"].to(device)
        all_data["test_labels"] = all_data["test_labels"].to(device)
        (Li, Ri, Lavgi) = train(new_net, all_data, hps, problem_params, opts, optimizer, saved_neurons, ferrouts, scaler, lang=lang, lpath=lpath)
        np.save("results/%s/data/saved_dynamics_d_%s_m_%s_%s_%s" % (problem_params["alias"], problem_params["d"], width, problem_params["name"], i), np.array(saved_neurons).astype(np.float16))
        np.save("results/%s/data/xout_d_%s_m_%s_%s_%s" % (problem_params["alias"], problem_params["d"], width, problem_params["name"], i), np.array(ferrouts))
        L = L + Li
        R = R + Ri
        Lavg = Lavg + Lavgi
    hps["epoch"] = epochs
    return saved_neurons, L, R, Lavg