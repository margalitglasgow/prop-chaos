import argparse
import prop_chaos
import plotting
import utils
import torch
import numpy as np

from matplotlib import pyplot as plt

# Set problem parameters and ground truth fn.
fresh_test_data = False
fresh_train_data = False
# Note: I found LR of 0.05 worked for IE 4 with d = 30...so trying to extrapolate based on some back-of-envelope theory here.
# Though sometimes this LR seems too big and there is oscilation...so may need to lower it (though this ofc slows down the training)

# Some teacher neurons that are useful in code below
e1 = np.zeros(2)
e1[0] = 1
e2 = np.zeros(2)
e2[1] = 1
ediag = np.zeros(2)
ediag[:2]= [1/np.sqrt(2), 1/np.sqrt(2)]
ediag2 = np.zeros(2)
ediag2[:2]= [-1/np.sqrt(2), 1/np.sqrt(2)]


hps = {"fresh_train": fresh_train_data, "fresh_test": fresh_test_data}

# Set up problem instances for SIM Problems.
teachers = torch.tensor([e1]).float()

PP = {}

problem_name = "He3"
k_GT = 3 # Information exponent
k_indices = 1
problem_params_He3 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He3

problem_name = "He4"
k_GT = 4 # Information exponent
k_indices = 1
problem_params_He4 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4

problem_name = "He4_small"
k_indices = 1
k_GT = 4 # Information exponent
her_dict_act_small = {4: 1}
her_dict_label_small = {4: 0.5}

poly_dict_act_small = utils.her_dic_to_pol(her_dict_act_small)
activation_small = utils.poly_act_fun(poly_dict_act_small)

poly_dict_label_small = utils.her_dic_to_pol(her_dict_label_small)
label_fun_small = utils.poly_label_fun(poly_dict_label_small)

problem_params_He4_small = {"activation": activation_small, "label_fun": label_fun_small, "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_small

problem_name = "He4_misspecified"
k_GT = 4 # Information exponent
k_indices = 1
her_dict_act_mis = {4: 1, 6: 1}
her_dict_label_mis = {4: 0.6, 6: 0.3}

poly_dict_act_mis = utils.her_dic_to_pol(her_dict_act_mis)
activation_mis = utils.poly_act_fun(poly_dict_act_mis)

poly_dict_label_mis = utils.her_dic_to_pol(her_dict_label_mis)
label_fun_mis = utils.poly_label_fun(poly_dict_label_mis)
problem_params_He4_misspecified = {"activation": activation_mis, "label_fun": label_fun_mis, "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_misspecified

problem_name = "He4_orth"
#2 Orthogonal Neurons ---------------
k_GT = 4 # Information exponent
k_indices = 2
teachers_orth = torch.tensor([e1, -e1, e2, -e2]).float()
problem_params_He4_orth = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_orth, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_orth

problem_name = "He4_nonorth"
#2 NonOrthogonal Neurons ---------------
k_GT = 4 # Information exponent
k_indices = 2
teachers_nonorth = torch.tensor([e1, -e1, ediag, -ediag]).float()
problem_params_He4_nonorth = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_nonorth, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_nonorth

problem_name = "He4_random_2_8"
# 8 random neurons in 2D
K = 8
k_GT = 4
k_indices = 2
teachers_rand = torch.zeros(size=(K,k_indices))
proj_rand = torch.normal(0,1, size=(K,k_indices))
teachers_rand[:, :k_indices] = proj_rand/torch.norm(proj_rand, dim=1, keepdim=True)
problem_params_He4_rand_2_8 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_rand, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_rand_2_8

problem_name = "He4_random_8_8"
# 8 random neurons in 8D
K = 8
k_GT = 4
k_indices = 8
teachers_rand2 = torch.zeros(size=(K,k_indices))
proj_rand2 = torch.normal(0,1, size=(K,k_indices))
teachers_rand2[:, :k_indices] = proj_rand2/torch.norm(proj_rand2, dim=1, keepdim=True)
problem_params_He4_rand_8_8 = {"activation": utils.her(k_GT), "label_fun": utils.teacher_fun(teachers_rand2, utils.her(k_GT)), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_He4_rand_8_8

problem_name = "Man_2"
# Target function is uniform over 2D circle
k_GT = 4
k_indices = 2
problem_params_man_2 = {"activation": utils.her(k_GT), "label_fun": utils.man_fun(k_indices), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_man_2

problem_name = "Man_big_2"
# Target function is uniform over 2D circle
k_GT = 4
k_indices = 2
problem_params_man_big_2 = {"activation": utils.her(k_GT), "label_fun": utils.man_big_fun(k_indices), "boolean": False, "signed": False, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_man_big_2

problem_name = "XOR_2"
# 2-XOR
k_GT = 2
k_indices = 2
problem_params_XOR_2 = {"activation": utils.relu, "label_fun": utils.XOR_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_2

problem_name = "XOR_3"
# 3-XOR
k_GT = 3
k_indices = 3
problem_params_XOR_3 = {"activation": utils.relu, "label_fun": utils.XOR_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_3

problem_name = "XOR_4"
# 4-XOR
k_GT = 4
k_indices = 4
problem_params_XOR_4 = {"activation": utils.relu, "label_fun": utils.XOR_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_4

problem_name = "XOR_bigg"
# 4-XOR
k_GT = 4
k_indices = 4
problem_params_XOR_4 = {"activation": utils.relu, "label_fun": utils.XOR_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_4

problem_name = "XOR_5"
# 5-XOR
k_GT = 5
k_indices = 5
problem_params_XOR_5 = {"activation": utils.relu, "label_fun": utils.XOR_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_XOR_5

problem_name = "Zero"
# Zero Function
k_GT = 0
k_indices = 4
problem_params_zero = {"activation": utils.relu, "label_fun": lambda x: torch.zeros(len(x)), "boolean": False, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_zero

problem_name = "Hop_4"
# XOR(x1..x4) + XOR(x1...x8) /2
k_GT = 4
k_indices = 8
problem_params_hop_4 = {"activation": utils.relu, "label_fun": utils.hopcase_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_hop_4

problem_name = "Hop_2"
# XOR(x1..x2) + XOR(x1...x4) /2
k_GT = 2
k_indices = 4
problem_params_hop_2 = {"activation": utils.relu, "label_fun": utils.hopcase_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_hop_2

problem_name = "Hop_2_4"
# XOR(x1..x2) + XOR(x1...x6) /2
k_GT = 4
k_indices = 6
problem_params_hop_2_4 = {"activation": utils.relu, "label_fun": utils.hop_fun(2, k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_hop_2_4

problem_name = "Hop_1_3"
# x1 + XOR(x1...x4) /2
k_GT = 3
k_indices = 4
problem_params_hop_1_3 = {"activation": utils.relu, "label_fun": utils.hop_fun(1, k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_hop_1_3

problem_name = "stair_3"
# x1 + x1x2 + x1x2x3
k_GT = 3
k_indices = 3
problem_params_stair_3 = {"activation": utils.relu, "label_fun": utils.staircase_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_stair_3

problem_name = "stair_4"
# x1 + x1x2 + x1x2x3 + x1x2x3x4
k_GT = 4
k_indices = 4
problem_params_stair_4 = {"activation": utils.relu, "label_fun": utils.staircase_fun(k_GT), "boolean": True, "signed": True, "corr": False, "name": problem_name, "k_GT": k_GT, "k_indices": k_indices}
PP[problem_name] = problem_params_stair_4

def complile_saved(d, widths, name, prefix):
    saved = []
    for M in widths:
        sn = np.load("results/%s/data/%s_d_%s_m_%s_%s.npy" % (name, prefix, d, M, name))
        saved.append(sn)
    saved = np.array(saved)
    np.save("results/%s/data/%s_d_%s_%s" % (name, prefix, d, name), saved)
    return saved

def run_sims(problem_params, d, widths, backup, hps, all_data, opts, stoc):
    name = problem_params["name"]
    max_width = widths[-1]
    for i in range(len(widths)):
        M = widths[i]
        print("Width: %s" % M)
        (sn, L, R, Lavg) = prop_chaos.init_and_train(M, backup, all_data, hps, problem_params, opts)
        sn_padded = np.zeros(shape=(int(hps["epoch"]), max_width, opts["k_indices"] + 1))
        sn_padded[:, :M, :] = np.array(sn)
        np.save("results/%s/data/saved_dynamics_d_%s_m_%s_%s" % (name, d, M, name), sn_padded)
        np.save("results/%s/data/losses_d_%s_m_%s_%s" % (name, d, M, name), L)
        np.save("results/%s/data/trainerr_d_%s_m_%s_%s" % (name, d, M, name), Lavg)
        np.save("results/%s/data/risks_d_%s_m_%s_%s" % (name, d, M, name), R)

def main():
    parser = argparse.ArgumentParser(description="Run Simulations.")
    parser.add_argument("-d", "--dimension", type=int, default=16)
    parser.add_argument("-mb", "--width_base", type=int, default=128)
    parser.add_argument("-ms", "--num_widths", type=int, default=3)
    parser.add_argument("-t", "--time", type=int, default=0)
    parser.add_argument("-p", "--problem", type=str, default="He4")
    parser.add_argument("-n", "--ndata", type=int, default=0)
    parser.add_argument("-l", "--lr", type=float, default=0.01)

    args = parser.parse_args()

    problem_params = PP[args.problem]
    
    d = args.dimension
    widths = []
    for i in range(args.num_widths):
        widths.append(args.width_base*(2**(i)))

    T = args.time
    if T == 0:
        T = max(np.log(d), 0.5*d**(problem_params["k_GT"]/2 - 1)) # Time (in MF scaling) = epoch * LR (I usually set this to 30 for k = 4, d = 30.)

    lr = args.lr
    
    hps["seed"] = 302
    if args.ndata == 0:
        n = 2048*d # Though this should perhaps scale up with d -- perhaps like d^k/2-1
    else:
        n = args.ndata
    batch = int(n*4/d) # If n: full-batch (no SGD)
    max_width = widths[-1]
    backup = torch.normal(0,1/np.sqrt(d), size=(max_width,d)) # All runs will agree upon this (in some subset)
    problem_params["d"] = d
    epoch = int(T*batch/(lr*n))
    hps["lr"] = lr
    hps["batch"] = batch
    hps["epoch"] = epoch
    hps["time"] = T
    stoc = int(n/batch)
    n_test = max(n, 2048)
    print("Name: %s, d: %s, n: %s, LR: %s, Time: %s, Epoch: %s" % (problem_params["name"], d, n, lr, epoch*lr*n/batch, epoch))

    opts = {"k_indices": problem_params["k_indices"], "print": False, "test_freq": 5}
    # For MIMs problems with > 2 indices, you can increase this, but the plotting becomes more difficult...can also keep this set to 2 and still use a higher-D target.

    # Fix training and test data set (this will only be used if we are training on the emp loss / we want a stable test set for smoother visualization)
    inputs, labels = prop_chaos.get_data(n, d, problem_params["label_fun"], boolean=problem_params["boolean"])
    test_inputs, test_labels = prop_chaos.get_data(n_test, d, problem_params["label_fun"], boolean=problem_params["boolean"])
    all_data = {"inputs": inputs, "labels": labels, "test_inputs": test_inputs, "test_labels": test_labels}

    name = problem_params["name"]
    try:
        back = np.load("results/%s/data/backup_d_%s_%s.npy" % (name, d, name))
        assert back.shape[0] == max_width
        backup = torch.from_numpy(back)
    finally:
        np.save("results/%s/data/backup_d_%s_%s" % (name, d, name), backup.detach().numpy())
        run_sims(problem_params, d, widths, backup, hps, all_data, opts, stoc)

    # Complies saved data
    for prefix in ["risks", "trainerr", "losses", "saved_dynamics"]:
        complile_saved(d, widths, name, prefix)

    # Make some plots
    # Plot LR Curves
    all_risks = np.load("results/%s/data/risks_d_%s_%s.npy" % (name, d, name))
    all_trains = np.load("results/%s/data/trainerr_d_%s_%s.npy" % (name, d, name))
    its = len(all_risks[0])
    T = hps["time"]
    ts = [(T/its)*j for j in range(its)]

    plotting.plot_LR(widths, ts, all_trains, all_risks, problem_params)

if __name__ == "__main__":
    main()