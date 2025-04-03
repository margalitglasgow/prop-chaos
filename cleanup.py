import problems
import prop_chaos
import plotting

import argparse
import utils
import torch
import numpy as np

def compute_diffs(d, widths, name, alias, k_indices):
    sn2 =  np.load("results/%s/data/saved_dynamics_d_%s_m_%s_%s.npy" % (alias, d, widths[-1], name))
    for i in range(len(widths) - 1):
        m = widths[i]
        sn1 = np.load("results/%s/data/saved_dynamics_d_%s_m_%s_%s.npy" % (alias, d, m, name))
        sn3 = sn2[:, :m, :]
        diff = sn1 - sn3
        normsq = np.square(np.linalg.norm(diff[:, :, :k_indices], axis=2))
        normsq = normsq +  np.square(diff[:, :, k_indices])*(d - k_indices)
        norm_all_d = np.sqrt(normsq)
        norm_k = np.linalg.norm(diff[:, :, :k_indices], axis=2)
        np.save("results/%s/plotdata/diffs_all_d_d_%s_m_%s_%s" % (alias, d, m, name), norm_all_d)
        np.save("results/%s/plotdata/diffs_k_d_%s_m_%s_%s" % (alias, d, m, name), norm_k)

def compute_ferr(d, widths, name, alias):
    fout2 =  np.load("results/%s/data/xout_d_%s_m_%s_%s.npy" % (alias, d, widths[-1], name))
    # Dimension should be #saves x n
    for i in range(len(widths) - 1):
        m = widths[i]
        fout1 = np.load("results/%s/data/xout_d_%s_m_%s_%s.npy" % (alias, d, m, name))
        diffsq = np.mean(np.square(fout2 - fout1), axis=1)
        np.save("results/%s/plotdata/function_err_d_%s_m_%s_%s" % (alias, d, m, name), diffsq)

def compile_saved(prefix, d, widths, name, alias):
    saved = []
    for M in widths:
        sn = np.load("results/%s/data/%s_d_%s_m_%s_%s.npy" % (alias, prefix, d, M, name))
        saved.append(sn)
    saved = np.array(saved)
    np.save("results/%s/data/%s_d_%s_%s" % (alias, prefix, d, name), saved)
    return saved

def compile_sn(prefix, d, widths, name, alias, div=1):
    for width in widths:
        for i in range(div):
            sni = np.load("results/%s/data/%s_d_%s_m_%s_%s_%s.npy" % (alias, prefix, d, width, name, i))
            print(sni.shape)
            if i == 0:
                saved = np.copy(sni)
            else:
                saved = np.concatenate((saved, sni))
        print("Final shaped of saved %s: %s" % (prefix, saved.shape))
        # (e, m, k) = saved.shape
        # sn_padded = np.zeros(shape=(e, widths[-1], k))
        # sn_padded[:, :m, :] = saved
        np.save("results/%s/data/%s_d_%s_m_%s_%s" % (alias, prefix, d, width, name), saved)

def main():
    parser = argparse.ArgumentParser(description="Run Simulations.")
    parser.add_argument("-d", "--dimension", type=int, default=16)
    parser.add_argument("-mb", "--width_base", type=int, default=128)
    parser.add_argument("-ms", "--num_widths", type=int, default=3)
    parser.add_argument("-t", "--time", type=int, default=0)
    parser.add_argument("-p", "--problem", type=str, default="He4")
    parser.add_argument("-v", "--div", type=int, default=1)
    parser.add_argument("-a", "--alias", type=str, default="")

    args = parser.parse_args()

    problem_params = problems.PP[args.problem]
    name = problem_params["name"]
    k_ind = problem_params["k_indices"]

    alias = args.alias
    if len(alias) == 0:
        alias = args.problem
    problem_params["alias"] = alias
    
    d = args.dimension
    widths = []
    for i in range(args.num_widths):
        widths.append(args.width_base*(2**(i)))
    problem_params["d"] = d

    T = args.time

    # Complies saved data into one file for each m
    for prefix in ["saved_dynamics", "xout"]:
        try:
            compile_sn(prefix, d, widths, name, alias, div=10)
        except FileNotFoundError:
            print("Skipping %s because could find files" % prefix)
        finally:
            pass

    # Compute and log norms of diffs (Delta_t(i))
    compute_diffs(d, widths, name, alias, k_ind)

    # Compute and log function error
    compute_ferr(d, widths, name, alias)

    # complile_saved("saved_dynamics", d, widths, name, alias)
    # for prefix in ["risks", "trainerr", "losses"]:
    #     complile_saved(prefix, d, widths, name, alias)

    # all_risks = np.load("results/%s/data/risks_d_%s_%s.npy" % (name, d, name))
    # all_trains = np.load("results/%s/data/trainerr_d_%s_%s.npy" % (name, d, name))
    # its = len(all_risks[0])
    # ts = [(T/its)*j for j in range(its)]
    # plotting.plot_LR(widths, ts, all_trains, all_risks, problem_params)


if __name__ == "__main__":
    main()