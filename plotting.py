import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np
import argparse
import seaborn as sns
from scipy.stats import gaussian_kde

def plot_all_LR(ds, widths, Ts, alias, name):
    color_cycle = plt.cm.tab10.colors  # or use plt.get_cmap('tab10') if you want more than 10
    num_colors = len(color_cycle)

    for d in ds:
        for i in range(len(widths)):
            M = widths[i]
            accum = 1
            # if M > 65536:
            #     accum = min(int(M / 65536), 16)
            try:
                Lavg = np.load(f'results/{alias}/data/trainerr_d_{d}_m_{M}_{name}.npy')
                R = np.load(f'results/{alias}/data/risks_d_{d}_m_{M}_{name}.npy')
            except FileNotFoundError:
                Lavg = np.load(f'results/{alias}/plotdata/trainerr_d_{d}_m_{M}_{name}.npy')
                R = np.load(f'results/{alias}/plotdata/risks_d_{d}_m_{M}_{name}.npy')

            time_per_save = Ts[0] / (len(Lavg) + 0.0)
            ts = [i * time_per_save for i in range(len(Lavg))]

            color = color_cycle[i % num_colors]
            plt.plot(ts, Lavg, label=f"Loss: m = {M}", color=color, linestyle='-')
            plt.plot(ts, R, label=f"Risk: m = {M}", color=color, linestyle='--')

    plt.legend(fontsize='small')
    plt.title("Loss and Risk")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.savefig(f"results/{alias}/LR_plot_{name}")
    plt.clf()
    plt.cla()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def plot_all_LR_big_width(ds, widths, Ts, alias, name):
    color_cycle = plt.cm.tab10.colors
    num_colors = len(color_cycle)

    # Store legend handles for experiment colors
    color_legend = []

    for i in range(len(ds)):
        d = ds[i]
        M = widths[-1]

        accum = 1
        # if M > 65536:
        #     accum = min(int(M / 65536), 16)

        Lavg = np.load(f'results/{alias}/plotdata/trainerr_d_{d}_m_{M}_{name}.npy')
        R = np.load(f'results/{alias}/plotdata/risks_d_{d}_m_{M}_{name}.npy')

        time_per_save = Ts[0] / (len(Lavg) + 0.0)
        ts = [i * time_per_save for i in range(len(Lavg))]

        color = color_cycle[i % num_colors]
        label = f"d={d}"

        # Plot Loss (solid) and Risk (dashed)
        plt.plot(ts, Lavg*accum, color=color, linestyle='-')
        plt.plot(ts, R, color=color, linestyle='--')

        # Add a single legend entry per experiment (use solid for display)
        color_legend.append(Line2D([0], [0], color=color, linestyle='-', label=label))

    # Add line style legend (black, once)
    style_legend = [
        Line2D([0], [0], color='black', linestyle='-', label='Loss'),
        Line2D([0], [0], color='black', linestyle='--', label='Risk')
    ]

    # Combine legends into one
    full_legend = style_legend + color_legend
    plt.legend(handles=full_legend, loc='upper right')

    plt.title("Loss and Risk")
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.savefig(f"results/{alias}/LR_plot_{name}")
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
    print("Starting Histograms")
    colors = sns.color_palette("tab10")
    #Plot Max
    # running_max = 1
    # for i in range(len(widths) - 1):
    #     M = widths[i]
    #     dataM = np.load('results/%s/plotdata/diffs_k_d_%s_m_%s_%s.npy' % (alias, d, M, name))
    #     data = np.max(dataM, axis=0)*np.sqrt(M)
    #     sns.histplot(data, kde=True, alpha=.3, color=colors[i], edgecolor=(1, 1, 1, .4), stat="density", binwidth=0.1, label="width: %s" % widths[i])
    #     # add quantiles
    #     quantiles_to_compute = [50, 75, 100*(1 - 1.0/np.sqrt(M))]
    #     quantiles = np.percentile(data, quantiles_to_compute)
    #     quantiles = quantiles.tolist()
    #     for quantile in quantiles:
    #         plt.axvline(quantile, color=colors[i], ymax=1.0, linestyle='--')
    #     running_max = np.maximum(running_max, 1.25*quantiles[-1])
    # plt.xlim(0, running_max)
    # plt.legend()
    # plt.title("Density of maximum parameter difference (times sqrt(m))")
    # plt.savefig("results/%s/Max_hist_d_%s_%s" % (alias, d, name))
    # plt.clf()
    # plt.cla()
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # two side-by-side plots
    running_max_scaled = 1
    running_max_raw = 1

    ymaxs = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for i in range(len(widths) - 1):
        M = widths[i]
        # if M < 10000:
        #     continue
        print(M)
        dataM = np.load(f'results/{alias}/plotdata/diffs_all_d_d_{d}_m_{M}_{name}.npy')
        #dataM = np.load(f'results/{alias}/plotdata/diffs_k_d_{d}_m_{M}_{name}.npy')
        raw_data = dataM[-1]
        data_max = np.max(dataM, axis=0)
        if len(raw_data) > 10000:
            raw_data = np.random.choice(raw_data, 10000, replace=False)
            data_max = np.random.choice(data_max, 10000, replace=False)

        data_max_scaled = data_max*np.sqrt(M)
        scaled_data = raw_data * np.sqrt(M)

        # Plot with sqrt(M) multiplier
        # sns.histplot(scaled_data, kde=True, alpha=0.3, color=colors[i],
        #             edgecolor=(1, 1, 1, .4), stat="density", binwidth=0.02,
        #             label=f"width: {M}", ax=axs[0, 0])
        
        # quantiles_scaled = np.percentile(scaled_data, [50, 75, 100*(1 - 1.0/np.sqrt(M))])
        # for q in quantiles_scaled:
        #     axs[0, 0].axvline(q, color=colors[i], linestyle='--', ymax=1.0)
        # running_max_scaled = max(running_max_scaled, 1.25 * quantiles_scaled[-1])

        for (data, a, b) in [(scaled_data, 0, 0), (raw_data, 0, 1), (data_max, 1, 1), (data_max_scaled, 1, 0) ]:
            sns.histplot(data, kde=False, alpha=0.3, color=colors[i],
                edgecolor=(1, 1, 1, .4), stat="density",
                label=f"width: {M}", ax=axs[a, b])
            kde = gaussian_kde(data, bw_method='scott')
            x_vals = np.linspace(min(data), max(data), 200)
            kde_vals = kde(x_vals)
            axs[a, b].plot(x_vals, kde_vals, color=colors[i])
            ymaxs[(a, b)] = max(max(kde_vals), ymaxs[(a, b)])
            axs[a, b].set_ylim(0, ymaxs[(a, b)]*1.3)

        # quantiles_raw = np.percentile(raw_data, [50, 75, 100*(1 - 1.0/np.sqrt(M))])
        # for q in quantiles_raw:
        #     axs[0, 1].axvline(q, color=colors[i], linestyle='--', ymax=1.0)
        # running_max_raw = max(running_max_raw, 1.25 * quantiles_raw[-1])

        # sns.histplot(data_max_scaled, kde=True, alpha=0.3, color=colors[i],
        #     edgecolor=(1, 1, 1, .4), stat="density", binwidth=0.02,
        #     label=f"width: {M}", ax=axs[1, 0])
        # kde = gaussian_kde(data_max_scaled, bw_method='scott')
        # x_vals = np.linspace(min(data_max_scaled), max(data_max_scaled), 200)
        # kde_vals = kde(x_vals)
        # axs[1, 0].plot(x_vals, kde_vals, color=colors[i])

        # sns.histplot(data_max, kde=False, alpha=0.3, color=colors[i],
        #     edgecolor=(1, 1, 1, .4), stat="density", binwidth=0.02,
        #     label=f"width: {M}", ax=axs[1, 1])
        # kde = gaussian_kde(data_max, bw_method='scott')
        # x_vals = np.linspace(min(data_max), max(data_max), 200)
        # kde_vals = kde(x_vals)
        # axs[1, 1].plot(x_vals, kde_vals, color=colors[i])

    # Final plot formatting
    # axs[0, 0].set_xlim(0, running_max_scaled)
    axs[0, 1].set_xlim(0, 100)
    axs[0, 0].set_title("Final parameter difference × √M")
    #axs[0, 0].legend()

    axs[0, 1].set_xlim(0, 1.0)
    # axs[0, 1].set_ylim(0, 0.5)
    axs[0, 1].set_title("Final parameter difference (no multiplier)")
    #axs[0, 1].legend()

    # axs[1, 0].set_xlim(0, running_max_scaled)
    axs[1, 0].set_xlim(0, 200)
    axs[1, 0].set_title("Max parameter difference × √M")
    #axs[1, 0].legend()

    axs[1, 1].set_xlim(0, 2.0)
    # axs[1, 1].set_ylim(0, 1.0)
    axs[1, 1].set_title("Max parameter difference (no multiplier)")
    axs[1, 1].legend()

    # for ax in axs:
    #     ax.set_ylabel("Density")
    #     ax.set_xlabel("Final parameter difference")

    plt.tight_layout()
    plt.savefig(f"results/{alias}/Hist_all_d_{d}_{name}.png")
    plt.clf()
    plt.cla()

def plot_omegahat(ds, widths, Ts, name, alias):
    for j in range(len(ds)):
        d = ds[j]
        for i in range(len(widths) - 1):
            M = widths[i]
            print(M)
            diffs_all_d = np.load('results/%s/plotdata/diffs_all_d_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            #diffs_all_d = np.load('results/%s/plotdata/diffs_k_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            diffs_all_d = diffs_all_d.astype(np.float64)
            omega_hat = np.mean(diffs_all_d, axis=1)
            time_per_save = Ts[j]/(len(omega_hat) + 0.0)
            ts = [i*time_per_save for i in range(len(omega_hat))]
            plt.plot(ts, (np.square(omega_hat))*M,  label="m = %s " % M )
        plt.title("Squared Mean parameter difference times width m")
        plt.legend()
        plt.savefig("results/%s/Omega_combined_M_d_%s_%s" % (alias, d, name))
        plt.clf()
        plt.cla()
    for j in range(len(ds)):
        d = ds[j]
        for i in range(len(widths) - 1):
            M = widths[i]
            print(M)
            diffs_all_d = np.load('results/%s/plotdata/diffs_all_d_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            #diffs_all_d = np.load('results/%s/plotdata/diffs_k_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            omega_hat = np.mean(diffs_all_d, axis=1)
            time_per_save = Ts[j]/(len(omega_hat) + 0.0)
            ts = [i*time_per_save for i in range(len(omega_hat))]
            plt.plot(ts, (np.square(omega_hat)),  label="m = %s " % M )
        plt.title("Squared Mean parameter difference")
        plt.legend()
        plt.savefig("results/%s/Omega_combined_d_%s_%s" % (alias, d, name))
        plt.clf()
        plt.cla()

    
def plot_omega(ds, widths, Ts, name, alias):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    for j in range(len(ds)):
        d = ds[j]
        for i in range(len(widths) - 1):
            M = widths[i]
            print(M)
            diffs_all_d = np.load('results/%s/plotdata/diffs_all_d_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            diffs_k = np.load('results/%s/plotdata/diffs_k_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            omega_k = np.mean(diffs_k, axis=1)
            omega_hat = np.mean(diffs_all_d, axis=1)
            print(len(omega_k))
            time_per_save = Ts[j]/(len(omega_k) + 0.0)
            ts = [i*time_per_save for i in range(len(omega_k))]
            axs[0, 0].plot(ts, np.square(omega_k)*M,  label="k: m = %s, d = %s " % (M, d) )
            axs[0, 1].plot(ts, np.square(omega_k),  label="k: m = %s, d = %s " % (M, d) )
            axs[1, 0].plot(ts, np.square(omega_hat)*M,  label="k: m = %s, d = %s " % (M, d) )
            axs[1, 1].plot(ts, np.square(omega_hat),  label="k: m = %s, d = %s " % (M, d) )
    plt.title("Squared Mean parameter difference (times width)")
    plt.legend()
    plt.savefig("results/%s/Omega_%s" % (alias, name))
    plt.clf()
    plt.cla()

def plot_function_err(ds, widths, Ts, name, alias):
    for j in range(len(ds)):
        d = ds[j]
        for i in range(len(widths) - 1):
            M = widths[i]
            ferr = np.load('results/%s/plotdata/function_err_d_%s_m_%s_%s.npy' % (alias, ds[j], M, name))
            ferr = ferr.astype(np.float64)
            time_per_save = Ts[j]/(len(ferr) + 0.0)
            ts = [i*time_per_save for i in range(len(ferr))]
            plt.plot(ts, ferr*M,  label="m = %s" % M )
        plt.title("Function Error times width m for d = %s" % d)
        plt.xlabel("Time")
        # plt.ylim(0, 1)
        plt.legend()
        plt.savefig("results/%s/Function_error_d_%s_%s" % (alias, d, name))
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
    
    d = args.dim
    ds = [d, 2*d, 4*d, 8*d]
    Ts = [args.time for i in range(len(ds))]

    #plot_all_LR(ds, widths, Ts, alias, name) # This is outdated
    plot_all_LR_big_width(ds, widths, Ts, alias, name)

    # plot_omega(ds, widths, Ts, name, alias) # Outdated
    plot_omegahat(ds, widths, Ts, name, alias)
    plot_function_err(ds, widths, Ts, name, alias)

    for d in ds:
        plot_histograms(d, widths, name, alias)

    # for width_index in range(len(widths) - 1):
    #     plot_individual_diffs(ds, widths, width_index, Ts, alias, name, args.kind)
    # for width_index in range(len(widths)):
    #     plot_alignment(ds, widths, width_index, Ts, alias, name, args.kind)


if __name__ == "__main__":
    main()


