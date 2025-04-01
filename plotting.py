import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import argparse
import seaborn as sns

def plot_LR(widths, ts, all_trains, all_risks, problem_params):
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
    name = problem_params['name']
    plt.savefig("results/%s/LR_plot_d_%s_%s" % (name, problem_params['d'], name))
    plt.clf()
    plt.cla()

def plot_histograms(d, widths, name):
    colors = sns.color_palette("tab10")
    saved= np.load('results/%s/data/saved_dynamics_d_%s_%s.npy' % (name, d, name))
    #Plot Max
    running_max = 1
    for i in range(len(widths) - 1):
        M = widths[i]
        comp = -1 # i + 1
        diff = np.array(saved[i])[:, :M, :] - np.array(saved[comp])[:, :M, :]

        data = np.max(np.linalg.norm(diff, axis=2), axis=0)*np.sqrt(M)
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
    plt.savefig("results/%s/Max_hist_d_%s_%s" % (name, d, name))
    plt.clf()
    plt.cla()
    
    #Plot histogram at last step
    running_max = 1
    for i in range(len(widths) - 1):
        M = widths[i]
        comp = -1 # i + 1
        diff = np.array(saved[i])[:, :M, :] - np.array(saved[comp])[:, :M, :]

        data = np.linalg.norm(diff, axis=2)[-1]*np.sqrt(M)
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
    plt.savefig("results/%s/Final_hist_d_%s_%s" % (name, d, name))
    plt.clf()
    plt.cla()
    
def plot_all_diffs(ds, widths, Ts, name, k_indices):
    for j in range(len(ds)):
        d = ds[j]
        saved= np.load('results/%s/data/saved_dynamics_d_%s_%s.npy' % (name, ds[j], name))
        time_per_save = Ts[j]/(len(saved[0]) + 0.0)
        ts = [i*time_per_save for i in range(len(saved[0]))]
        for i in range(len(saved) - 1):
            M = widths[i]
            comp = -1 # i + 1
            diff = np.array(saved[i])[:, :M, :] - np.array(saved[comp])[:, :M, :]
            normsq = np.square(np.linalg.norm(diff[:, :, :k_indices], axis=2))
            normsq = normsq +  np.square(diff[:, :, k_indices])*(d - k_indices)
            norm = np.sqrt(normsq)
            omega = np.mean(norm, axis=1)
            #omega = np.mean(np.linalg.norm(diff, axis=2), axis=1) # This is average norm of diffs (in 2D space)
            plt.plot(ts, np.square(omega)*M,  label="m = %s, d = %s " % (M, d) )
    plt.title("Squared Mean parameter difference (times width)")
    plt.legend()
    plt.savefig("results/%s/Omega_squared_%s" % (name, name))
    plt.clf()
    plt.cla()
    for j in range(len(ds)):
        d = ds[j]
        all_risks = np.load('results/%s/data/risks_d_%s_%s.npy' % (name, ds[j], name))
        time_per_save = Ts[j]/len(all_risks[0])
        ts = [i*time_per_save for i in range(len(all_risks[0]))]
        for i in range(len(all_risks) - 1):
            M = widths[i]
            comp = -1 # i + 1
            R_diff = np.abs(np.array(all_risks[i]) - np.array(all_risks[comp]))*M
            plt.plot(ts, R_diff, label="m = %s, d = % s" % (M, d))
    plt.title("Risk Diffs")
    plt.legend()
    plt.savefig("results/%s/Risk_diffs_%s" % (name, name))
    plt.clf()
    plt.cla()
    for j in range(len(ds)):
        d = ds[j]
        all_losses = np.load('results/%s/data/losses_d_%s_%s.npy' % (name, ds[j], name))
        time_per_save = Ts[j]/len(all_losses[0])
        ts = [i*time_per_save for i in range(len(all_losses[0]))]
        for i in range(len(all_losses) - 1):
            M = widths[i]
            comp = -1 # i + 1
            L_diff = np.abs((np.array(all_losses[i]) - np.array(all_losses[comp])))*M
            plt.plot(ts, L_diff, label="m = %s, d = %s " % (M, d) )
    plt.title("Training Loss Diffs (per tr step)")
    plt.legend()
    plt.savefig("results/%s/Loss_diffs_%s" % (name, name))
    plt.clf()
    plt.cla()
    for j in range(len(ds)):
        d = ds[j]
        all_trains = np.load('results/%s/data/trainerr_d_%s_%s.npy' % (name, ds[j], name))
        time_per_save = Ts[j]/len(all_trains[0])
        ts = [i*time_per_save for i in range(len(all_trains[0]))]
        for i in range(len(all_trains) - 1):
            M = widths[i]
            comp = -1 # i + 1
            L_diff = np.abs((np.array(all_trains[i]) - np.array(all_trains[comp])))*M
            plt.plot(ts, L_diff, label="m = %s, d = %s " % (M, d) )
    plt.title("Training Loss Diffs (Per Epoch)")
    plt.legend()
    plt.savefig("results/%s/TrainingLoss_diffs_%s" % (name, name))
    plt.clf()
    plt.cla()

def plot_individual_diffs(ds, widths, width_index, Ts, name, k_indices):
    for j in range(len(ds)):
        saved= np.load('results/%s/data/saved_dynamics_d_%s_%s.npy' % (name, ds[j], name))
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
            plt.plot(ts, normsq)
        normsq = np.square(np.linalg.norm(diff[:, :, :k_indices], axis=2))
        normsq = normsq +  np.square(diff[:, :, k_indices])*(ds[j] - k_indices)
        norm = np.sqrt(normsq)
        omega = np.mean(norm, axis=1)
        #omega = np.mean(np.linalg.norm(diff, axis=2), axis=1) # This is average norm of diffs (in 2D space)
        plt.plot(ts, omega[:], linewidth=4, c="black", label=r'$\mathbb{E}_i \|\Delta_t(i)\|$')
        plt.savefig("results/%s/Individual_diffs_d_%s_m_%s_%s" % (name, ds[j], M, name))
        plt.clf()
        plt.cla()

def plot_alignment(ds, widths, width_index, Ts, name, k_indices):
    # ts = [i*hps["lr"] for i in range(len(SN))]
    for j in range(len(ds)):
        saved= np.load('results/%s/data/saved_dynamics_d_%s_%s.npy' % (name, ds[j], name))
        time_per_save = Ts[j]/(len(saved[0]) + 0.0)
        ts = [i*time_per_save for i in range(len(saved[0]))]
        M = widths[width_index]
        SN = saved[width_index][:, :M, :]
        alphas = np.linalg.norm(SN[:, :, :k_indices], axis=2)
        for i in range(len(SN[0])):
            plt.plot(ts, alphas[:, i])
        plt.xlabel("Time")
        plt.ylabel(r'Alignment $\alpha_t(w)$')
        plt.savefig("results/%s/Alignment_d_%s_m_%s_%s" % (name, ds[j], M, name))
        plt.clf()
        plt.cla()

def main():
    parser = argparse.ArgumentParser(description="Make Plots.")
    parser.add_argument("-p", "--problem", type=str, default="He4")
    parser.add_argument("-mb", "--width_base", type=int, default=128)
    parser.add_argument("-ms", "--num_widths", type=int, default=3)
    parser.add_argument("-t", "--time", type=int, default=0)
    parser.add_argument("-k", "--kind", type=int, default=1)
    args = parser.parse_args()
    name = args.problem

    widths = []
    for i in range(args.num_widths):
        widths.append(args.width_base*(2**(i)))
    
    ds = [64, 128]
    #Ts = [16, 16, 32, 64] #[200] #[100, 400, 1600, 3200]
    Ts = [args.time for i in range(len(ds))]

    plot_all_diffs(ds, widths, Ts, name, args.kind)
    width_index = 0
    for width_index in range(len(widths) - 1):
        plot_individual_diffs(ds, widths, width_index, Ts, name, args.kind)
    for width_index in range(len(widths)):
        plot_alignment(ds, widths, width_index, Ts, name, args.kind)

    for d in ds:
        plot_histograms(d, widths, name)

if __name__ == "__main__":
    main()

