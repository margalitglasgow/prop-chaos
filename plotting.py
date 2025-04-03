import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import argparse
import seaborn as sns

def plot_all_LR(ds, widths, Ts, alias, name):
    for d in ds:
        for i in range(len(widths)):
            M = widths[i]
            try:
                Lavg = np.load('results/%s/data/trainerr_d_%s_m_%s_%s.npy' % (alias, d, M, name))
                R = np.load('results/%s/data/risks_d_%s_m_%s_%s.npy' % (alias, d, M, name))
            except FileNotFoundError:
                Lavg = np.load('results/%s/plotdata/trainerr_d_%s_m_%s_%s.npy' % (alias, d, M, name))
                R = np.load('results/%s/plotdata/risks_d_%s_m_%s_%s.npy' % (alias, d, M, name))
            time_per_save = Ts[0]/(len(Lavg) + 0.0)
            ts = [i*time_per_save for i in range(len(Lavg))]
            plt.plot(ts, Lavg, label="Loss: width %s, d %s" % (M, d))
            plt.plot(ts, R, label="Risk: width %s, d %s" % (M, d))
    plt.legend()
    plt.savefig("results/%s/LR_plot_%s" % (alias, name))
    plt.clf()
    plt.cla()

def plot_LR(widths, ts, all_trains, all_risks, name, alias):
    for i in range(len(widths)):
        M = widths[i]
        ord = 2*(len(widths)-i)
        Lavg = all_trains[i]
        R = all_risks[i]
        # Plot losses and risks
        plt.plot(ts, Lavg, label="Loss: width %s" % M, zorder=ord)
        plt.plot(ts, R, label="Risk: width %s" % M, zorder=ord)
        # plt.plot(L, label="Loss: width %s" % M, zorder=ord)
        # plt.plot([len(L)/len(R)*i for i in range(len(R))], R, label="Risk: width %s" % M, zorder=ord+1)
    plt.legend()
    plt.savefig("results/%s/LR_plot_d_%s_%s" % (alias, d, name))
    plt.clf()
    plt.cla()

def plot_histograms(d, widths, name, alias):
    colors = sns.color_palette("tab10")
    #Plot Max
    running_max = 1
    for i in range(len(widths) - 1):
        M = widths[i]
        dataM = np.load('results/%s/plotdata/diffs_k_d_%s_m_%s_%s.npy' % (alias, d, M, name))
        data = np.max(dataM, axis=0)*np.sqrt(M)
        sns.histplot(data, kde=True, alpha=.3, color=colors[i], edgecolor=(1, 1, 1, .4), stat="density", binwidth=0.1, label="width: %s" % widths[i])
        # add quantiles
        quantiles_to_compute = [50, 75, 100*(1 - 1.0/np.sqrt(M))]
        quantiles = np.percentile(data, quantiles_to_compute)
        quantiles = quantiles.tolist()
        for quantile in quantiles:
            plt.axvline(quantile, color=colors[i], ymax=1.0, linestyle='--')
        running_max = np.maximum(running_max, 1.25*quantiles[-1])
    plt.xlim(0, running_max)
    plt.legend()
    plt.title("Density of maximum parameter difference (times sqrt(m))")
    plt.savefig("results/%s/Max_hist_d_%s_%s" % (alias, d, name))
    plt.clf()
    plt.cla()
    
    #Plot histogram at last step
    running_max = 1
    for i in range(len(widths) - 1):
        M = widths[i]
        dataM = np.load('results/%s/plotdata/diffs_k_d_%s_m_%s_%s.npy' % (alias, d, M, name))
        data = dataM[-1]*np.sqrt(M)
        sns.histplot(data, kde=True, alpha=.3, color=colors[i], edgecolor=(1, 1, 1, .4), stat="density", binwidth=0.02, label="width: %s" % widths[i])
    
        # add quantiles
        quantiles_to_compute = [50, 75, 100*(1 - 1.0/np.sqrt(M))]
        quantiles = np.percentile(data, quantiles_to_compute)
        quantiles = quantiles.tolist()
        for quantile in quantiles:
            plt.axvline(quantile, color=colors[i], ymax=1.0, linestyle='--')
        running_max = np.maximum(running_max, 1.25*quantiles[-1])
    plt.xlim(0, running_max)
    plt.title("Density of final parameter difference (times sqrt(m))")
    plt.legend()
    plt.savefig("results/%s/Final_hist_d_%s_%s" % (alias, d, name))
    plt.clf()
    plt.cla()
    
def plot_omega(ds, widths, Ts, name, alias):
    for j in range(len(ds)):
        d = ds[j]
        for i in range(len(widths) - 1):
            M = widths[i]
            diffs_k = np.load('results/%s/plotdata/diffs_k_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            diffs_all_d = np.load('results/%s/plotdata/diffs_all_d_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            omega_k = np.mean(diffs_k, axis=1)
            omega_hat = np.mean(diffs_all_d, axis=1)

            time_per_save = Ts[j]/(len(omega_k) + 0.0)
            ts = [i*time_per_save for i in range(len(omega_k))]
            plt.plot(ts, np.square(omega_k)*M,  label="k: m = %s, d = %s " % (M, d) )
            plt.plot(ts, np.square(omega_hat)*M,  label="hat: m = %s, d = %s " % (M, d) )
    plt.title("Squared Mean parameter difference (times width)")
    plt.legend()
    plt.savefig("results/%s/Omega_squared_%s" % (alias, name))
    plt.clf()
    plt.cla()

def plot_function_err(ds, widths, Ts, name, alias):
    for j in range(len(ds)):
        d = ds[j]
        for i in range(len(widths) - 1):
            M = widths[i]
            ferr = np.load('results/%s/plotdata/function_err_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            time_per_save = Ts[j]/(len(ferr) + 0.0)
            ts = [i*time_per_save for i in range(len(ferr))]
            plt.plot(ts, ferr*M,  label="m = %s, d = %s " % (M, d) )
    plt.title("Function Error (times width)")
    plt.legend()
    plt.savefig("results/%s/Function_error_%s" % (alias, name))
    plt.clf()
    plt.cla()

def plot_individual_diffs(ds, widths, width_index, Ts, name, k_indices):
    for j in range(len(ds)):
        saved= np.load('results/%s/data/saved_dynamics_d_%s_%s.npy' % (alias, ds[j], name))
        time_per_save = Ts[j]/(len(saved[0]) + 0.0)
        ts = [i*time_per_save for i in range(len(saved[0]))]
        M = widths[width_index]
        sn1 = saved[width_index][:, :M, :]
        sn2 = saved[-1][:, :M, :]#saved[width_index + 1][:, :M, :]
        diff = sn1 - sn2
        for i in range(M):
            normsq = np.square(np.linalg.norm(diff[:, i, :k_indices], axis=1))
            normsq = normsq + np.square(diff[:, i, k_indices])*(ds[j] - k_indices)
            norm = np.sqrt(normsq)
            norm = np.linalg.norm(diff[:, i, :k_indices], axis=1)
            plt.plot(ts, norm)
        normsq = np.square(np.linalg.norm(diff[:, :, :k_indices], axis=2))
        normsq = normsq +  np.square(diff[:, :, k_indices])*(ds[j] - k_indices)
        norm = np.sqrt(normsq)
        omega = np.mean(norm, axis=1)
        #omega = np.mean(np.linalg.norm(diff, axis=2), axis=1) # This is average norm of diffs (in 2D space)
        plt.plot(ts, omega[:], linewidth=4, c="black", label=r'$\mathbb{E}_i \|\Delta_t(i)\|$')
        plt.savefig("results/%s/Individual_diffs_d_%s_m_%s_%s" % (alias, ds[j], M, name))
        plt.clf()
        plt.cla()

def plot_alignment(ds, widths, width_index, Ts, name, k_indices):
    # ts = [i*hps["lr"] for i in range(len(SN))]
    for j in range(len(ds)):
        saved= np.load('results/%s/data/saved_dynamics_d_%s_%s.npy' % (alias, ds[j], name))
        time_per_save = Ts[j]/(len(saved[0]) + 0.0)
        ts = [i*time_per_save for i in range(len(saved[0]))]
        M = widths[width_index]
        SN = saved[width_index][:, :M, :]
        alphas = np.linalg.norm(SN[:, :, :k_indices], axis=2)
        for i in range(len(SN[0])):
            plt.plot(ts, alphas[:, i])
        plt.xlabel("Time")
        plt.ylabel(r'Alignment $\alpha_t(w)$')
        plt.savefig("results/%s/Alignment_d_%s_m_%s_%s" % (alias, ds[j], M, name))
        plt.clf()
        plt.cla()

def main():
    parser = argparse.ArgumentParser(description="Make Plots.")
    parser.add_argument("-p", "--problem", type=str, default="He4")
    parser.add_argument("-a", "--alias", type=str, default="")
    parser.add_argument("-mb", "--width_base", type=int, default=128)
    parser.add_argument("-ms", "--num_widths", type=int, default=3)
    parser.add_argument("-t", "--time", type=int, default=0)
    parser.add_argument("-k", "--kind", type=int, default=1)
    parser.add_argument("-d", "--dim", type=int, default=32)
    args = parser.parse_args()
    name = args.problem
    alias = args.alias
    if len(alias) == 0:
        alias = name

    widths = []
    for i in range(args.num_widths):
        widths.append(args.width_base*(2**(i)))
    
    ds = [args.dim]
    Ts = [args.time for i in range(len(ds))]

    #plot_omega(ds, widths, Ts, name, alias)
    plot_function_err(ds, widths, Ts, name, alias)
    # for d in ds:
    #     plot_histograms(d, widths, name, alias)


    # for width_index in range(len(widths) - 1):
    #     plot_individual_diffs(ds, widths, width_index, Ts, alias, name, args.kind)
    # for width_index in range(len(widths)):
    #     plot_alignment(ds, widths, width_index, Ts, alias, name, args.kind)


    #plot_all_LR(ds, widths, Ts, alias, name)

if __name__ == "__main__":
    main()


