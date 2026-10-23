import csv
import numpy as np
from pathlib import Path
import QDL
import experiments

N = 6       # number of message qubits
K = 4       # number of keys (2 classical bits)
SEED = 7    # seed for rng
d = 2**N    # Hilbert space dimension
DIR = Path(__file__).resolve().parent.parent   # the Project folder (main.py is in Project/src)


def summary_table(metrics: dict) -> str:
    """Format the averaged recovery probabilities and entropies as a table.
        Example:
        scenario       recovery  entropy [bits]
        correct key    1.000000          0.0000
        wrong key      0.006098          5.3527
        no key         0.001616          5.4275
        direct         0.015625          6.0000
    """
    
    rows = [("correct key", "correct"), ("wrong key", "wrong"),
            ("no key", "nokey"), ("direct", "direct")]
    lines = [f"{'scenario':<12} {'recovery':>10} {'entropy [bits]':>15}"]
    
    for label, name in rows:
        lines.append(f"{label:<12} {metrics[name + '_recovery']:>10.6f} "
                     f"{metrics[name + '_entropy']:>15.4f}")

    return "\n".join(lines)


def mutual_info_table(metrics: dict, M: int) -> str:
    """Format the mutual information I(m; mp) for each decoding strategy as a table.
        The last column is the fraction of the maximum, log2(M) bits for M tested messages.
        Example (n=6, K=4, seed 7):
        strategy       I(m;mp) [bits]  of max
        correct key            6.0000   1.000
        guessed key            0.8953   0.149
        no key                 0.1361   0.023
        direct                 0.0000   0.000
    """

    max_info = np.log2(M)   # recover all info
    rows = [("correct key", "correct"), ("guessed key", "guess"),
            ("no key", "nokey"), ("direct", "direct")]
    lines = [f"{'strategy':<12} {'I(m;mp) [bits]':>16} {'of max':>7}"]

    for label, name in rows:
        info = metrics["mutual_info_" + name]
        lines.append(f"{label:<12} {info:>16.4f} {info / max_info:>7.3f}")

    return "\n".join(lines)


def sweep_table(sweep_results: list[dict]) -> str:
    """Format the parameter sweep over n and K as a table, one row per (n, K).
        M = number of messages tested, rec. = average recovery probability, H = average entropy.
        Example:
         n  K    M  correct rec.  wrong rec.       1/d  wrong H [bits]  no-key H [bits]
         4  2   16      1.000000    0.026929  0.062500          3.5641           3.2897
         4  4   16      1.000000    0.118225  0.062500          3.5288           3.3878
         6  2   64      1.000000    0.015421  0.015625          5.2146           5.4435
         6  4   64      1.000000    0.006098  0.015625          5.3527           5.4275
         8  2   32      1.000000    0.005834  0.003906          7.4294           7.4656
         8  4   32      1.000000    0.002547  0.003906          7.3909           7.4252
    """

    # header row, OBS: column widths match the value rows below
    lines = [f"{'n':>2} {'K':>2} {'M':>4} {'correct rec.':>13} {'wrong rec.':>11} {'1/d':>9} "
             f"{'wrong H [bits]':>15} {'no-key H [bits]':>16}"]

    # one row per (n, K) combination from experiments.parameter_sweep
    for r in sweep_results:
        n, K = r['n'], r['K']
        d = 2**n
        # 1/d = probability of guessing the message blindly, to compare wrong rec. against
        lines.append(f"{n:>2} {K:>2} {r['M']:>4} {r['correct_recovery']:>13.6f}"
                     f"{r['wrong_recovery']:>11.6f} {1 / d:>9.6f} "
                     f"{r['wrong_entropy']:>15.4f} {r['nokey_entropy']:>16.4f}")

    return "\n".join(lines)

def print_resources() -> None:
    """Print number of gates for QFT, Dk, full circuit, and depth of full circuit, for n = 4, 6, 8."""
    for n in [4, 6, 8]:
        r = experiments.circuit_resources(n, K, SEED)
        print(f"n={n}: QFT {r['QFT']} gates, Dk {r['Dk']} gates, "
              f"full {r['full']} gates, depth {r['depth']}")



def main():
    # Test all messages for n <= 6, or a fixed random subset of 32 for larger n
    msgs = experiments.generate_messages(N, seed=SEED)
    M = len(msgs)
    # Test all available keys (0 to K-1)
    keys = list(range(K))

    # ----------- Metrics for chosen n, K ------------------------------------------------
    phases, batch_probabilities = experiments.start_experiment(msgs, keys, N, K, SEED)
    metrics = experiments.analyze_results(batch_probabilities, K, msgs, keys)
    
    # Print/save table of metrics summary for N qubits, K keys
    table = summary_table(metrics)
    print(f"n={N}, d={d}, K={K}, seed={SEED}")
    print(table)
    (DIR / "results").mkdir(exist_ok=True)
    (DIR / f"results/summary_n{N}_K{K}_seed{SEED}.txt").write_text(table)

    # Print/save table of mutual information for each decoding strategy
    info_table = mutual_info_table(metrics, M)
    print()
    print(info_table)
    (DIR / f"results/mutual_info_n{N}_K{K}_seed{SEED}.txt").write_text(info_table)

    # Save one row to CSV for each decoding attempt (message, correct key, attempted key, result)
    experiments.save_trials(batch_probabilities, msgs, keys, DIR / "results", f"trials_n{N}_K{K}_seed{SEED}.csv")

    # ----------- Binary phases ------------------------------------------------
    # Same experiment with phases restricted to 0 or pi (meaning diagonal entries of Dk are +1 or -1)
    binary_phases, binary_probabilities = experiments.start_experiment(msgs, keys, N, K, SEED, binary=True)
    binary_metrics = experiments.analyze_results(binary_probabilities, K, msgs, keys)

    # Print/save table of metrics summary for the binary codebook
    binary_table = summary_table(binary_metrics)
    print()
    print(f"n={N}, d={d}, K={K}, seed={SEED}, binary phases")
    print(binary_table)
    (DIR / f"results/summary_n{N}_K{K}_seed{SEED}_binary.txt").write_text(binary_table)

    binary_info_table = mutual_info_table(binary_metrics, M)
    print()
    print(binary_info_table)
    (DIR / f"results/mutual_info_n{N}_K{K}_seed{SEED}_binary.txt").write_text(binary_info_table)

    # ----------- Plotting ------------------------------------------------
    # For a given message, key: plot probabilities each output state when 
    # using correct key, no key, averaged wrong key, and direct measurement
    m_idx = 2   # index in msgs of sent message
    k_idx = 1   # index in keys of correct key
    m, k = msgs[m_idx], keys[k_idx]
    experiments.plot_mk_probabilities(batch_probabilities, m_idx, k_idx, msgs, keys, K,
                                      DIR / "results/plots", f"distribution_m{m}_k{k}.png")

    # same (message, key) pair for the binary codebook
    experiments.plot_mk_probabilities(binary_probabilities, m_idx, k_idx, msgs, keys, K,
                                      DIR / "results/plots", f"distribution_m{m}_k{k}_binary.png")

    # ------------- Circuit diagrams --------------------------------------------
    # one diagram with right key
    QDL.draw_example(m, k, k, phases, N, DIR / "results/plots/circuit_correct_key.png")
    # one diagram with wrong key
    QDL.draw_example(m, k, (k + 1) % K, phases, N, DIR / "results/plots/circuit_wrong_key.png")

    # ----------- Sweep params ------------------------------------------------
    # get metrics whens sweeping over different n, K
    # sweep_results is a list of dicts
    sweep_results = experiments.parameter_sweep([4, 6, 8], [2, 4], SEED)
    
    # Print/save table of the sweep
    print()
    print(sweep_table(sweep_results))

    with open(DIR / f"results/sweep_seed{SEED}.csv", "w", newline="") as f:
        # list(sweep_results[0]) gets keys of the sweep_result rows
        writer = csv.DictWriter(f, fieldnames=list(sweep_results[0]))
        writer.writeheader()
        writer.writerows(sweep_results)
            

    return metrics


if __name__ == '__main__':
    metrics = main()
    #print_resources()