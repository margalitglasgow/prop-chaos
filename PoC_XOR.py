#!/usr/bin/env python3
import argparse
import math
import time
import pickle

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learning k-parity via spherical SGD"
    )
    # Data and training parameters
    parser.add_argument("--d", type=int, default=32,
                        help="Input dimension")
    parser.add_argument("--train_size", type=int, default=2**16,
                        help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=2**14,
                        help="Mini-batch size (also for test)")
    parser.add_argument("--T0", type=float, default=200,
                        help="Time")
    parser.add_argument("--eta0", type=float, default=0.02,
                        help="Base learning rate")
    parser.add_argument("--num_experiments", type=int, default=3,
                        help="Number of subnetwork experiments")
    parser.add_argument("--widths", type=int, nargs='+',
                        default=[2**12, 2**13, 2**14, 2**15, 2**17],
                        help="List of network widths (small to large)")
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Use automatic mixed precision")
    parser.add_argument("--save_result", action="store_true", default=True,
                        help="Whether to save results to pickle")
    parser.add_argument("--save_filename", type=str, default="XOR4.pkl",
                        help="Filename for saving results")
    parser.add_argument("--chunk_size", type=int, default=1024,
                        help="Chunk size for forward/backward pass")
    parser.add_argument("--record_frequency", type=int, default=5,
                        help="Frequency (in steps) to record metrics")
    parser.add_argument("--verbose", action="store_true", default=False, 
                        help="Print progress")
    return parser.parse_args()


# Constants
k = 4
temperature = 16
cst = 2**k / math.sqrt(k)


def generate_data(n, d):
    """
    Generate n training samples in {-1,1}^d and the k-parity target.
    The target is 3/4 * parity(first k coords) + 1/4 * first bit.
    """
    X = (2 * torch.randint(0, 2, (n, d), device=device) - 1).float()
    y = X[:, :k].prod(dim=1).float()
    # y = 3/4 * y + 1/4 * X[:, 0]
    return X, y


def activation(x):
    return F.softplus(x, beta=temperature)


def activation_prime(x):
    return torch.sigmoid(temperature * x)


def forward_pass(weights, x, chunk_size=1024, use_chunking=True):
    N = weights.shape[0]
    B = x.shape[0]
    # build fixed second-layer v
    v = torch.empty(N, device=weights.device, dtype=weights.dtype)
    n_pos = N // 2
    v[:n_pos] = cst / N
    v[n_pos:] = -cst / N
    out = torch.zeros(B, device=weights.device, dtype=weights.dtype)
    if use_chunking and N > chunk_size:
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            w_chunk = weights[i:end]
            v_chunk = v[i:end]
            z = x @ w_chunk.t()
            out += (activation(z) * v_chunk).sum(dim=1)
    else:
        z = x @ weights.t()
        out = (activation(z) * v).sum(dim=1)
    return out


def train_step(weights, x, y, eta0, chunk_size=1024, use_chunking=True):
    N, d = weights.shape
    B = x.shape[0]
    # build fixed second-layer v to be plus-minus one
    v = torch.empty(N, device=weights.device, dtype=weights.dtype)
    n_pos = N // 2
    v[:n_pos] = cst / N
    v[n_pos:] = -cst / N
    f_accum = torch.zeros(B, device=weights.device, dtype=weights.dtype)
    chunk_info = []
    if use_chunking and N > chunk_size:
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            w_chunk = weights[start:end]
            z = x @ w_chunk.t()
            chunk_info.append((start, end, z))
            v_chunk = v[start:end]
            f_accum += (activation(z) * v_chunk).sum(dim=1)
    else:
        z = x @ weights.t()
        chunk_info.append((0, N, z))
        f_accum = (activation(z) * v).sum(dim=1)
    f_out = f_accum
    loss = ((f_out - y)**2).mean()
    error = f_out - y
    effective_eta = 2 * eta0 * N / cst / B
    with torch.no_grad():
        for start, end, z in chunk_info:
            v_chunk = v[start:end]
            multiplier = activation_prime(z) * error.unsqueeze(1) * v_chunk.unsqueeze(0)
            grad_chunk = multiplier.t() @ x
            updated = weights[start:end] - effective_eta * grad_chunk
            weights[start:end].copy_(F.normalize(updated, p=2, dim=1))
    return loss.item(), f_out.detach()


def evaluate_network(weights, x, chunk_size=1024, use_chunking=True):
    with torch.no_grad():
        return forward_pass(weights, x, chunk_size, use_chunking)


def main(args):
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.verbose:
        print(f"Using device: {device}")
        
    T = int(args.T0 / args.eta0)
    widths = args.widths
    print(widths)
    N_max = widths[-1]
    verbose_interval = max(int(T // 100), 1)
    
    # seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # init teacher
    W_max = F.normalize(torch.randn(N_max, args.d, device=device), p=2, dim=1)
    W_max_init = W_max.clone()
    largest_network = W_max.clone()
    metrics_largest = {'train_loss': [], 'test_loss': []}
    experiments = {}
    exp_mappings = {}
    metrics_exps = {}
    for exp in range(args.num_experiments):
        experiments[exp] = {}
        exp_mappings[exp] = {}
        metrics_exps[exp] = {}
        for w in widths[:-1]:
            half = w // 2
            pos_idx = torch.randperm(N_max // 2, device=device)[:half]
            neg_idx = torch.randperm(N_max - N_max//2, device=device)[:half] + N_max//2
            mapping = torch.cat([pos_idx, neg_idx])
            exp_mappings[exp][w] = mapping
            experiments[exp][w] = W_max_init[mapping].clone()
            metrics_exps[exp][w] = {k: [] for k in ['train_loss','test_loss','output_diff','neuron_diff']}
            
    # data
    X_train, y_train = generate_data(args.train_size, args.d)
    X_test_t, y_test_t = generate_data(args.batch_size, args.d)
    X_diff_t, _ = generate_data(args.batch_size, args.d)
    exp_test = {}
    exp_diff = {}
    for exp in range(args.num_experiments):
        exp_test[exp] = generate_data(args.batch_size, args.d)
        exp_diff[exp] = generate_data(args.batch_size, args.d)
        
    # initial metrics
    idx0 = torch.randperm(args.train_size, device=device)[:args.batch_size]
    xb0, yb0 = X_train[idx0], y_train[idx0]
    tr0 = forward_pass(largest_network, xb0, args.chunk_size, True)
    metrics_largest['train_loss'].append(((tr0 - yb0)**2).mean().item())
    te0 = evaluate_network(largest_network, X_test_t, args.chunk_size, True)
    metrics_largest['test_loss'].append(((te0 - y_test_t)**2).mean().item())
    for exp in range(args.num_experiments):
        X_te_e, y_te_e = exp_test[exp]
        X_df_e, _ = exp_diff[exp]
        for w in widths[:-1]:
            net = experiments[exp][w]
            tr = forward_pass(net, xb0, args.chunk_size, w>args.chunk_size)
            metrics_exps[exp][w]['train_loss'].append(((tr - yb0)**2).mean().item())
            te = evaluate_network(net, X_te_e, args.chunk_size, w>args.chunk_size)
            metrics_exps[exp][w]['test_loss'].append(((te - y_te_e)**2).mean().item())
            pd = forward_pass(net, X_df_e, args.chunk_size, w>args.chunk_size)
            td = forward_pass(largest_network, X_df_e, args.chunk_size, True)
            metrics_exps[exp][w]['output_diff'].append(((td - pd).pow(2).sum().item() / args.batch_size) * w)
            with torch.no_grad():
                dist = (net - largest_network[exp_mappings[exp][w]]).pow(2).sum(dim=1).mean().item() * w
            metrics_exps[exp][w]['neuron_diff'].append(dist)
            
    if args.verbose:
        print("Recorded initial statistics at T=0.")
        
    # training
    start = time.time()
    for step in range(T):
        idx = torch.randperm(args.train_size, device=device)[:args.batch_size]
        xb, yb = X_train[idx], y_train[idx]
        if args.use_amp:
            with torch.cuda.amp.autocast():
                loss_t, _ = train_step(largest_network, xb, yb, args.eta0, args.chunk_size, True)
        else:
            loss_t, _ = train_step(largest_network, xb, yb, args.eta0, args.chunk_size, True)
        for exp in range(args.num_experiments):
            for w in widths[:-1]:
                _ = train_step(experiments[exp][w], xb, yb, args.eta0, args.chunk_size, w>args.chunk_size)
        if (step+1) % args.record_frequency == 0:
            te = evaluate_network(largest_network, X_test_t, args.chunk_size, True)
            test_l = ((te - y_test_t)**2).mean().item()
            metrics_largest['train_loss'].append(loss_t)
            metrics_largest['test_loss'].append(test_l)
            for exp in range(args.num_experiments):
                X_te_e, y_te_e = exp_test[exp]
                X_df_e, _ = exp_diff[exp]
                for w in widths[:-1]:
                    net = experiments[exp][w]
                    te_w = evaluate_network(net, X_te_e, args.chunk_size, w>args.chunk_size)
                    metrics_exps[exp][w]['test_loss'].append(((te_w - y_te_e)**2).mean().item())
                    metrics_exps[exp][w]['train_loss'].append(loss_t)
                    pd = forward_pass(net, X_df_e, args.chunk_size, w>args.chunk_size)
                    td = forward_pass(largest_network, X_df_e, args.chunk_size, True)
                    metrics_exps[exp][w]['output_diff'].append(((td - pd).pow(2).sum().item() / args.batch_size)*w)
                    with torch.no_grad():
                        avg_d = (experiments[exp][w] - largest_network[exp_mappings[exp][w]]).norm(dim=1).mean().item()
                        metrics_exps[exp][w]['neuron_diff'].append((avg_d**2)*w)
            if (step+1) % verbose_interval == 0:
                if args.verbose:
                    print(f"Step {step+1}: Teacher Loss = {loss_t:.6f}, Test Loss = {test_l:.6f}")
                for exp in range(args.num_experiments):
                    if args.verbose:
                        print("-----------------------------------------------------")
                    for w in widths[:-1]:
                        tr_l = metrics_exps[exp][w]['train_loss'][-1]
                        te_l = metrics_exps[exp][w]['test_loss'][-1]
                        od = metrics_exps[exp][w]['output_diff'][-1]
                        nd = metrics_exps[exp][w]['neuron_diff'][-1]
                        if args.verbose:
                            print(f"  Exp {exp+1}, Width {w}: Train Loss = {tr_l:.6f}, Test Loss = {te_l:.6f}, Output Diff = {od:.6f}, Neuron Diff = {nd:.6f}")
                            
        torch.cuda.empty_cache()
        
    elapsed = time.time() - start
    if args.verbose:
        print(f"\nTraining completed in {elapsed:.2f} seconds.")
    
    if args.save_result:
        results = {
            'metrics_largest': metrics_largest,
            'metrics_experiments': metrics_exps,
            'hyperparameters': vars(args)
        }
        with open(args.save_filename, 'wb') as f:
            pickle.dump(results, f)
        print("Results saved")


if __name__ == "__main__":
    args = parse_args()
    main(args)
