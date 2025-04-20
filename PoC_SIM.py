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
        description="Train two-layer network via spherical SGD"
    )
    # Data and training parameters
    parser.add_argument("--d", type=int, default=32,
                        help="Input dimension")
    parser.add_argument("--train_size", type=int, default=2**16,
                        help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=2**13,
                        help="Mini-batch size (also for test)")
    parser.add_argument("--T0", type=float, default=15,
                        help="Time")
    parser.add_argument("--eta0", type=float, default=0.01,
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
    parser.add_argument("--save_filename", type=str, default="results.pkl",
                        help="Filename for saving results")
    parser.add_argument("--chunk_size", type=int, default=1024,
                        help="Chunk size for forward/backward pass")
    parser.add_argument("--record_frequency", type=int, default=1,
                        help="Frequency (in steps) to record metrics")
    parser.add_argument("--verbose", action="store_true", default=False, 
                        help="Print progress")
    return parser.parse_args()


# Normalized Hermite functions up to degree 6

def He4(x):
    return (x**4 - 6 * x**2 + 3) / math.sqrt(24)

def He4_prime(x):
    return (4 * x**3 - 12 * x) / math.sqrt(24)

def He6(x):
    return (x**6 - 15 * x**4 + 45 * x**2 - 15) / math.sqrt(720)

def He6_prime(x):
    return (6 * x**5 - 60 * x**3 + 90 * x) / math.sqrt(720)


def generate_data(n, d, theta, device):
    X = torch.randn(n, d, device=device)
    y = He4(X @ theta)
    return X, y


def activation(x):
    return He4(x)


def activation_prime(x):
    return He4_prime(x)


def forward_pass(weights, x, chunk_size=1024, use_chunking=True):
    N = weights.shape[0]
    out = torch.zeros(x.shape[0], device=weights.device)
    if use_chunking and N > chunk_size:
        for i in range(0, N, chunk_size):
            w_chunk = weights[i : i + chunk_size]
            out += activation(x @ w_chunk.t()).sum(dim=1)
    else:
        out = activation(x @ weights.t()).sum(dim=1)
    return out / N


def train_step(weights, x, y, eta0, chunk_size=1024, use_chunking=True):
    N = weights.shape[0]
    B = x.shape[0]
    # Forward pass with chunking info
    f_accum = torch.zeros(B, device=weights.device)
    chunk_info = []
    if use_chunking and N > chunk_size:
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            w_chunk = weights[start:end]
            z = x @ w_chunk.t()
            chunk_info.append((start, end, z))
            f_accum += activation(z).sum(dim=1)
    else:
        z = x @ weights.t()
        chunk_info.append((0, N, z))
        f_accum = activation(z).sum(dim=1)

    f_out = f_accum / N
    loss = ((f_out - y)**2).mean()
    error = f_out - y
    effective_eta = 2 * eta0 / B

    # Backward spherical gradient
    with torch.no_grad():
        for start, end, z in chunk_info:
            grad_chunk = (activation_prime(z) * error.unsqueeze(1)).t() @ x
            updated = weights[start:end] - effective_eta * grad_chunk
            weights[start:end].copy_(F.normalize(updated, p=2, dim=1))

    return loss.item(), f_out.detach()


def evaluate_network(weights, x, chunk_size=1024, use_chunking=True):
    with torch.no_grad():
        return forward_pass(weights, x, chunk_size, use_chunking)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.verbose:
        print(f"Using device: {device}")

    # Hyperparams
    T = int(args.T0 / args.eta0)
    widths = args.widths
    N_max = widths[-1]
    verbose_interval = max(int(T // 100), 1)

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Teacher direction
    theta = torch.randn(args.d, device=device)
    theta = F.normalize(theta, p=2, dim=0)

    # Initialize largest network
    W_max_init = F.normalize(torch.randn(N_max, args.d, device=device), p=2, dim=1)
    largest_network = W_max_init.clone()

    # Metric storage
    metrics_largest = {'train_loss': [], 'test_loss': []}
    metrics_exps = {}
    exp_mappings = {}
    experiments = {}

    # Setup experiments
    for exp in range(args.num_experiments):
        exp_mappings[exp] = {}
        experiments[exp] = {}
        metrics_exps[exp] = {}
        for w in widths[:-1]:
            mapping = torch.randperm(N_max)[:w]
            exp_mappings[exp][w] = mapping
            experiments[exp][w] = W_max_init[mapping].clone()
            metrics_exps[exp][w] = {k: [] for k in ['train_loss', 'test_loss', 'output_diff', 'neuron_diff']}

    # Generate data
    X_train, y_train = generate_data(args.train_size, args.d, theta, device)
    X_test_large, y_test_large = generate_data(args.batch_size, args.d, theta, device)
    X_diff_large, _ = generate_data(args.batch_size, args.d, theta, device)

    exp_test = {}
    exp_diff = {}
    for exp in range(args.num_experiments):
        exp_test[exp] = generate_data(args.batch_size, args.d, theta, device)
        exp_diff[exp] = generate_data(args.batch_size, args.d, theta, device)

    # Initial statistics
    idx0 = torch.randperm(args.train_size)[:args.batch_size]
    xb0, yb0 = X_train[idx0], y_train[idx0]
    # Teacher
    pred_train = forward_pass(largest_network, xb0, args.chunk_size, True)
    metrics_largest['train_loss'].append(((pred_train - yb0)**2).mean().item())
    pred_test = evaluate_network(largest_network, X_test_large, args.chunk_size, True)
    metrics_largest['test_loss'].append(((pred_test - y_test_large)**2).mean().item())

    # Subnetworks initial
    for exp in range(args.num_experiments):
        X_test_e, y_test_e = exp_test[exp]
        X_diff_e, _ = exp_diff[exp]
        for w in widths[:-1]:
            net = experiments[exp][w]
            tr = forward_pass(net, xb0, args.chunk_size, w>args.chunk_size)
            metrics_exps[exp][w]['train_loss'].append(((tr - yb0)**2).mean().item())
            te = evaluate_network(net, X_test_e, args.chunk_size, w>args.chunk_size)
            metrics_exps[exp][w]['test_loss'].append(((te - y_test_e)**2).mean().item())
            pd = forward_pass(net, X_diff_e, args.chunk_size, w>args.chunk_size)
            td = forward_pass(largest_network, X_diff_e, args.chunk_size, True)
            metrics_exps[exp][w]['output_diff'].append(((td - pd).pow(2).sum().item() / args.batch_size) * w)
            dist = (net - largest_network[exp_mappings[exp][w]]).pow(2).sum(dim=1).mean().item() * w
            metrics_exps[exp][w]['neuron_diff'].append(dist)

    if args.verbose:
        print("Recorded initial statistics at T=0.")

    # Training loop
    start = time.time()
    for step in range(T):
        idx = torch.randperm(args.train_size)[:args.batch_size]
        xb, yb = X_train[idx], y_train[idx]
        # Teacher step
        if args.use_amp:
            with torch.cuda.amp.autocast():
                loss_large, _ = train_step(largest_network, xb, yb, args.eta0,
                                           args.chunk_size, True)
        else:
            loss_large, _ = train_step(largest_network, xb, yb, args.eta0,
                                       args.chunk_size, True)

        # Subnetwork steps
        for exp in range(args.num_experiments):
            for w in widths[:-1]:
                net = experiments[exp][w]
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss_w, _ = train_step(net, xb, yb, args.eta0,
                                               args.chunk_size, w>args.chunk_size)
                else:
                    loss_w, _ = train_step(net, xb, yb, args.eta0,
                                           args.chunk_size, w>args.chunk_size)
                if (step + 1) % args.record_frequency == 0:
                    metrics_exps[exp][w]['train_loss'].append(loss_w)

        # Record metrics
        if (step + 1) % args.record_frequency == 0:
            te_large = evaluate_network(largest_network, X_test_large, args.chunk_size, True)
            test_loss_large = ((te_large - y_test_large)**2).mean().item()
            metrics_largest['train_loss'].append(loss_large)
            metrics_largest['test_loss'].append(test_loss_large)

            for exp in range(args.num_experiments):
                X_test_e, y_test_e = exp_test[exp]
                X_diff_e, _ = exp_diff[exp]
                for w in widths[:-1]:
                    net = experiments[exp][w]
                    te = evaluate_network(net, X_test_e, args.chunk_size, w>args.chunk_size)
                    metrics_exps[exp][w]['test_loss'].append(((te - y_test_e)**2).mean().item())
                    pd = forward_pass(net, X_diff_e, args.chunk_size, w>args.chunk_size)
                    td = forward_pass(largest_network, X_diff_e, args.chunk_size, True)
                    metrics_exps[exp][w]['output_diff'].append(((td - pd).pow(2).sum().item() / args.batch_size) * w)
                    dist = (net - largest_network[exp_mappings[exp][w]]).norm(dim=1).mean().item()**2 * w
                    metrics_exps[exp][w]['neuron_diff'].append(dist)

            # Verbose logging
            if (step + 1) % verbose_interval == 0:
                if args.verbose:
                    print(f"\nStep {step+1:4d} Statistics:")
                    print(f"Teacher network: Train Loss = {loss_large:.6f}, Test Loss = {test_loss_large:.6f}")
                for exp in range(args.num_experiments):
                    if args.verbose:
                        print("-----------------------------------------------------")
                    for w in widths[:-1]:
                        tr_loss = metrics_exps[exp][w]['train_loss'][-1]
                        te_loss = metrics_exps[exp][w]['test_loss'][-1]
                        out_diff = metrics_exps[exp][w]['output_diff'][-1]
                        n_diff = metrics_exps[exp][w]['neuron_diff'][-1]
                        if args.verbose:
                            print(f"Exp {exp+1} | Width {w:7d}: Train Loss = {tr_loss:.6f}, Test Loss = {te_loss:.6f}, "
                                  f"Output Diff (scaled) = {out_diff:.6f}, Neuron Diff (scaled) = {n_diff:.6f}")

        torch.cuda.empty_cache()

    elapsed = time.time() - start
    if args.verbose:
        print(f"\nTraining completed in {elapsed:.2f} seconds.")

    # Save results
    if args.save_result:
        out = {
            'metrics_largest': metrics_largest,
            'metrics_experiments': metrics_exps,
            'hyperparameters': vars(args)
        }
        with open(args.save_filename, 'wb') as f:
            pickle.dump(out, f)
        print("Results saved")


if __name__ == "__main__":
    args = parse_args()
    main(args)