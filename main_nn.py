import prop_chaos
import utils
import problems

import argparse
import torch
# from torch.cuda.amp import autocast, GradScaler
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import logging

def run_sims(problem_params, d, widths, backup, hps, all_data, opts, lang=0, lpath=0):
    name = problem_params["name"]
    alias = problem_params["alias"]
    max_width = widths[-1]
    for i in range(len(widths)):
        M = widths[i]
        logging.info("Starting sim with width: %s" % M)
        (sn, L, R, Lavg) = prop_chaos.init_and_train(M, backup, all_data, hps, problem_params, opts, lang=lang, lpath=lpath)
        logging.info("Ending sim with width: %s" % M)
        np.save("results/%s/data/losses_d_%s_m_%s_%s" % (alias, d, M, name), L)
        # np.save("results/%s/data/trainerr_d_%s_m_%s_%s" % (alias, d, M, name), Lavg)
        # np.save("results/%s/data/risks_d_%s_m_%s_%s" % (alias, d, M, name), R)
        np.save("results/%s/plotdata/trainerr_d_%s_m_%s_%s" % (alias, d, M, name), Lavg)
        np.save("results/%s/plotdata/risks_d_%s_m_%s_%s" % (alias, d, M, name), R)    
        
    # for prefix in ["risks", "trainerr", "losses"]:
    #     compile_saved(d, widths, name, alias, prefix)
    # Make some plots
    # Plot LR Curves
    # all_risks = np.load("results/%s/data/risks_d_%s_%s.npy" % (alias, d, name))
    # all_trains = np.load("results/%s/data/trainerr_d_%s_%s.npy" % (alias, d, name))
    # its = len(all_risks[0])
    # T = hps["time"]
    # ts = [(T/its)*j for j in range(its)]
    # plotting.plot_LR(widths, ts, all_trains, all_risks, problem_params)

        # sn_padded = np.zeros(shape=(int(hps["epoch"]), max_width, opts["k_indices"] + 1))
        # sn_padded[:, :M, :] = np.array(sn)
        # np.save("results/%s/data/saved_dynamics_d_%s_m_%s_%s" % (name, d, M, name), sn_padded)

def main():
    parser = argparse.ArgumentParser(description="Run Simulations.")
    parser.add_argument("-d", "--dimension", type=int, default=16)
    parser.add_argument("-mb", "--width_base", type=int, default=128)
    parser.add_argument("-ms", "--num_widths", type=int, default=3)
    parser.add_argument("-t", "--time", type=int, default=0)
    parser.add_argument("-p", "--problem", type=str, default="He4")
    parser.add_argument("-a", "--alias", type=str, default="")
    parser.add_argument("-n", "--ndata", type=int, default=0)
    parser.add_argument("-l", "--lr", type=float, default=0.01)
    parser.add_argument("-v", "--langevin", type=float, default=0)
    parser.add_argument("-r", "--pathreg", type=float, default=0)
    parser.add_argument("-j", "--jobid", type=int, default=0)
    args = parser.parse_args()

    problem_params = problems.PP[args.problem]
    hps = {"fresh_train": False, "fresh_test": False}
    opts = {"k_indices": problem_params["k_indices"], "print": False, "test_freq": 2}

    name = problem_params["name"]

    alias = args.alias
    if len(alias) == 0:
        alias = args.problem
    problem_params["alias"] = alias

    d = args.dimension
    problem_params["d"] = d

    widths = []
    for i in range(args.num_widths):
        widths.append(args.width_base*(2**(i)))
    max_width = widths[-1]

    T = args.time
    if T == 0:
        T = max(np.log(d), 0.5*d**(problem_params["k_GT"]/2 - 1)) # Time (in MF scaling) = epoch * LR (I usually set this to 30 for k = 4, d = 30.)
    hps["time"] = T

    lr = args.lr
    hps["lr"] = lr
    
    hps["seed"] = 302
    if args.ndata == 0:
        n = 2048*d # Though this should perhaps scale up with d -- perhaps like d^k/2-1
    else:
        n = args.ndata
    batch = int(n*4/d) # If n: full-batch (no SGD)
    hps["batch"] = batch

    epoch = int(T*batch/(lr*n))
    hps["epoch"] = epoch

    n_test = max(n, 2048)

    # Set up logger
    logging.basicConfig(filename="results/%s/info_%s.log" % (alias, args.jobid), level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Name: %s, d: %s, n: %s, LR: %s, Time: %s, Epoch: %s" % (problem_params["name"], d, n, lr, epoch*lr*n/batch, epoch))
    # np.random.seed(None)
    # st0 = np.random.get_state()
    # np.random.set_state(st0)
    # st1 = torch.initial_seed()
    # torch.manual_seed(st1)
    # logging.info("Numpy seed: %s, Torch seed: %s" % (st0, st1))

    # Load or create training and test data
    try:
        inputs_np = np.load("results/%s/data/inputs_d_%s_%s.npy" % (alias, d, name))
        labels_np = np.load("results/%s/data/labels_d_%s_%s.npy" % (alias, d, name))
        test_inputs_np = np.load("results/%s/data/test_inputs_d_%s_%s.npy" % (alias, d, name))
        test_labels_np = np.load("results/%s/data/test_labels_d_%s_%s.npy" % (alias, d, name))
        inputs = torch.from_numpy(inputs_np)
        labels = torch.from_numpy(labels_np)
        test_inputs = torch.from_numpy(test_inputs_np)
        test_labels = torch.from_numpy(test_labels_np)
        logging.info("Loaded Train and Test Data")
    except FileNotFoundError:
        logging.info("Couldn't find existing data so using new data")
        inputs, labels = prop_chaos.get_data(n, d, problem_params["label_fun"], boolean=problem_params["boolean"])
        test_inputs, test_labels = prop_chaos.get_data(n_test, d, problem_params["label_fun"], boolean=problem_params["boolean"])
        np.save("results/%s/data/inputs_d_%s_%s" % (alias, d, name), inputs.detach().numpy())
        np.save("results/%s/data/labels_d_%s_%s" % (alias, d, name), labels.detach().numpy())
        np.save("results/%s/data/test_inputs_d_%s_%s" % (alias, d, name), test_inputs.detach().numpy())
        np.save("results/%s/data/test_labels_d_%s_%s" % (alias, d, name), test_labels.detach().numpy())
    finally:
        all_data = {"inputs": inputs, "labels": labels, "test_inputs": test_inputs, "test_labels": test_labels}
    # Load or create backup net
    try:
        back = np.load("results/%s/data/backup_d_%s_%s.npy" % (alias, d, name))
        logging.info("Loaded Backup")
        m_back = back.shape[0]
        backup = torch.from_numpy(back)
        # Make backup bigger if needed
        if m_back < max_width:
            new_backup = torch.normal(0,1/np.sqrt(d), size=(max_width,d)) # All runs will agree upon this (in some subset)
            new_backup[:m_back, :] = backup
            backup = new_backup
            logging.info("Extended Backup from size %s to size %s" % (m_back, max_width))
            np.save("results/%s/data/backup_d_%s_%s" % (alias, d, name), backup.detach().numpy())
    except FileNotFoundError:
        logging.info("Couldn't find existing backup so making new backup")
        backup = torch.normal(0,1/np.sqrt(d), size=(max_width,d)) # All runs will agree upon this (in some subset)
        np.save("results/%s/data/backup_d_%s_%s" % (alias, d, name), backup.detach().numpy())
    finally:
        run_sims(problem_params, d, widths, backup, hps, all_data, opts)

    # Complies saved data
    # compile_sn("saved_dynamics", d, widths, name, div=10)
    # compile_sn("xout", d, widths, name, div=10)
    # for prefix in ["risks", "trainerr", "losses", "saved_dynamics"]:
    #     compile_saved(d, widths, name, prefix)

if __name__ == "__main__":
    main()