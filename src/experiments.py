import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from pathlib import Path
import csv
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate


import QDL as dlock


## Probability decoding 
def decoded_probabilities(m: int, k: int, kp: int | None, phases: np.ndarray, n: int) -> np.ndarray:
    """
    Get probabilities of one simulated lock-decode circuit,
    Args:
        m: message sent
        k: secret key
        kp: key (attempted) to unlock with
        phases: sampled codebook phases
        n: number of qubits
    Returns:
        probs: vector of d probabilies probs[i] of measuring msg i of decoded state vector
    """

    qc = dlock.lock(m, k, phases, n)
    qc.compose(dlock.decode(kp, phases, n), inplace=True)

    # instead of simulating for some num of shots,
    # get probabilities exactly from state vector
    decoded_sv = Statevector.from_instruction(qc)
    probs = decoded_sv.probabilities()

    return probs

def direct_probabilities(m: int, k: int, phases: np.ndarray, n: int) -> np.ndarray:
    """
    Get probabilities of measuring the locked state directly, without any decoding.
    Args:
        m: message sent
        k: secret key
        phases: sampled codebook phases
        n: number of qubits
    Returns:
        probs: vector of d probabilies probs[i] of measuring basis state i of the locked state vector
    """

    qc = dlock.lock(m, k, phases, n)

    # instead of simulating for some num of shots,
    # get probabilities exactly from state vector
    sv = Statevector.from_instruction(qc)
    return sv.probabilities()

# Entropy calculation
def calculate_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Calculate the Shannon entropy (in bits) of a probability distribution.
    
    Args:
        probs: A 1D array of probabilities, or a 2D array (K+2, d) from mk_pair_experiment.
        
    Returns:
        The entropy value(s). If a 2D array is passed, returns a 1D array of entropies 
        calculated along the last axis.
    """

    # np.log2 evaluates to -inf for 0. We use `where=(probs!=0)` and `out=np.zeros_like` 
    # to safely force 0 * log2(0) to equal 0, avoiding runtime warnings and NaNs.
    return -np.sum(probs * np.log2(probs, out=np.zeros_like(probs), where=(probs!=0)), axis=-1)


def generate_messages(n: int, M: int = 32, seed: int = 7) -> list[int]:
    """
    Generate a list of messages to test.
    For n <= 6, return all possible messages (0 to 2**n - 1).
    For n > 6, return a reproducible random subset of M messages.
    """

    d = 2**n
    # small enough to test every message
    if n <= 6:
        return list(range(d))

    # fixed seed, so the same M messages are picked every run
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(d, M, replace=False).tolist())


# sweep function to provide results for different values of K and n
def parameter_sweep(n_values: list[int], K_values: list[int], seed: int = 7):
    """
    Perform a parameter sweep over different values of n and K, running experiments for each combination.
    
    Args:
        n_values: List of qubit counts to test.
        K_values: List of key counts to test.
        seed: Random seed for reproducibility.
    Returns:
        sweep_results: list of dicts, one for each (n,K) pair, with keys
            ["n", "K", "M", "correct_recovery", "wrong_recovery", "wrong_entropy",
              "nokey_entropy", "direct_recovery"]
            M is the number of messages tested (all 2**n for n <= 6, a subset for larger n)
    """

    sweep_results = []

    for n in n_values:
        for K in K_values:
            msgs = generate_messages(n, M=32, seed=seed)
            keys = list(range(K))

            phases, batch_probs = start_experiment(msgs, keys, n, K, seed)
            metrics = analyze_results(batch_probs, K, msgs, keys)

            sweep_results.append({
                "n": n,
                "K": K,
                "M": len(msgs),     # number of messages tested (all for n <= 6)
                "correct_recovery": metrics["correct_recovery"],
                "wrong_recovery": metrics["wrong_recovery"],
                "wrong_entropy": metrics["wrong_entropy"],
                "nokey_entropy": metrics["nokey_entropy"],
                "direct_recovery": metrics["direct_recovery"]
            })

    return sweep_results


# Experiment functions
def start_experiment(msgs: list[int], keys: list[int], n: int, K: int,
                      seed: int = 7, binary: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Start the experiment by generating phases and running batch experiments with averaged probabilities 
    for correct key and wrong key cases and entropyies for no-key and wrong-key cases. 
    
    Args:
        msgs: list of messages to test
        keys: list of keys to test
        n: number of qubits
        K: number of keys
        seed: random seed for reproducibility
        binary: if True, sample the phases from {0, pi} instead of [0, 2pi)
    Returns:
        phases: 2D array of shape (K, 2**n) containing the generated phases.
        batch_probs: 4D array of shape (len(msgs), len(keys), K+2, 2**n) containing the probabilities for each message-key pair.
    """

    phases = dlock.sample_phases(n, K, seed, binary)
    batch_probs = mk_batch_experiment(msgs, keys, phases, n)
    return phases, batch_probs


# message key experiment functions 
def mk_pair_experiment(m: int, k:int, phases: np.ndarray, n: int) -> np.ndarray:
    """
    For message m, key k, get the correct-key, wrong-key, no-key, and
    direct measurement probability distributions.

    Args:
        m: message to test
        k: key to test
        phases: sampled codebook phases
        n: number of qubits
    Returns:
        probs: (K+2, d) shape: K wrong/correct key distributions, 1 no-key distribution, 1 direct measurement distribution
    """

    K, d = phases.shape
    probs = np.zeros((K+2,d))
    for kp in range(K):
        probs[kp,:] = decoded_probabilities(m, k, kp, phases, n)
        
    probs[K, :] = decoded_probabilities(m, k, None, phases, n) # no-key decoding
    probs[K+1, :] = direct_probabilities(m, k, phases, n) # direct measurement

    return probs


def mk_batch_experiment(msgs: list[int], keys: list[int], phases: np.ndarray, n: int) -> np.ndarray: 
    """
    For every (m,k) pair, prepare the locked state and test decoding with:
    the correct key k, every incorrect key kp, no phase correction, and direct measurement.
    
    Returning a 4D array of probabilities for such a batch of experiments.

    Args:
        msgs: list of messages to test
        keys: list of keys to test
        phases: sampled codebook phases
        n: number of qubits

    Returns:
        batch_probs: 4D array of shape (len(msgs), len(keys), K+2, d) 
                     containing probabilities for each message-key pair.
    """

    K, d = phases.shape
    batch_probs = np.zeros((len(msgs), len(keys), K + 2, d))

    for i, m in enumerate(msgs):
        for j, k in enumerate(keys):
            batch_probs[i, j] = mk_pair_experiment(m, k, phases, n)

    return batch_probs


def analyze_results(batch_probs: np.ndarray, K: int, msgs: list[int], keys: list[int]) -> dict:
    """
    Extracts the recovery probability for the sent message and averages the per-trial 
    entropies to prevent distribution smearing. 

    Args:
        batch_probs: 4D array of shape (len(msgs), len(keys), K+2, 2**n) containing
                     the probabilities for each message-key pair.
        K: number of keys
        msgs: list of messages to test
        keys: list of keys to test
        
    Returns:
        metrics: dictionary containing the average probabilities, entropies and mutual information for correct key,
                 wrong key, no key, and direct measurement cases.
    """

    # calculate entropy for each single trial first (Shape: len(msgs), len(keys), K+2)
    trial_entropies = calculate_entropy(batch_probs)
    recovery_probs = np.zeros((len(msgs), len(keys), K + 2))

    for i, m in enumerate(msgs):
        for j, k in enumerate(keys):
            for kp in range(K + 2):
                # extract probability specifically at index 'm' for sent message
                recovery_probs[i, j, kp] = batch_probs[i, j, kp, m]

    # aggregate correct-key scenarios
    correct_recovery = np.mean([recovery_probs[i, j, keys[j]] for i in range(len(msgs)) for j in range(len(keys))])
    correct_entropy = np.mean([trial_entropies[i, j, keys[j]] for i in range(len(msgs)) for j in range(len(keys))])

    wrong_recovery = np.mean([recovery_probs[i, j, kp] for i in range(len(msgs)) for j in range(len(keys)) for kp in range(K) if kp != keys[j]])
    wrong_entropy = np.mean([trial_entropies[i, j, kp] for i in range(len(msgs)) for j in range(len(keys)) for kp in range(K) if kp != keys[j]])

    nokey_recovery = np.mean(recovery_probs[:, :, K])
    nokey_entropy = np.mean(trial_entropies[:, :, K])

    direct_recovery = np.mean(recovery_probs[:, :, K + 1])
    direct_entropy = np.mean(trial_entropies[:, :, K + 1])

    # ---- Calculating mutual information: ----
    # batch_probs[m, k, kp, mp] is P(mp | m, k) when decoding with decoding strategy kp
    # To get mutual information for one decoding strategy, test Eve blindly using that strategy
    # Extract the probabilities for that strategy, and average across true keys (axis 1)
    # -> fixed_decoding_probs[m, kp, mp] is P(mp | m) when decoding with that strategy kp (specific key guess, no key, direct)

    fixed_decoding_probs = batch_probs.mean(axis=1)

    # get mutual info for guessing a key kp, averaged over all key guesses
    mutual_info_per_kp = [mutual_info_fixed_decoding(fixed_decoding_probs[:, kp])
                                 for kp in range(K)]
    mutual_info_guess = np.mean(mutual_info_per_kp)

    # get mutual info for no-key strategy
    mutual_info_nokey = mutual_info_fixed_decoding(fixed_decoding_probs[:,K])
    # get mutual info for direct measurement
    mutual_info_direct = mutual_info_fixed_decoding(fixed_decoding_probs[:,K+1])

    # -- Get Bobs mutual info (correct key) for reference (should be logM bits since he has right key) --
    # First get a list of K probability batches P(mp | m, k) when decoding with kp = k (the correct key),
    # one for each correct key k
    # bob_probs_per_k[k][m, mp] = P(mp | m, k) when decoding with kp = k
    bob_probs_per_k = [batch_probs[:, k, k] for k in range(K)] 
    # Then marginalize those over correct keys k
    # bob_probs[m, mp] = P(mp | m)
    bob_probs = np.mean(bob_probs_per_k, axis=0) # P(k) = 1/K so we take mean
    # Then get mutual info
    mutual_info_correct = mutual_info_fixed_decoding(bob_probs)
    # ----
    # ----------------------------------------------

    metrics = {
        "correct_recovery": correct_recovery,
        "correct_entropy": correct_entropy,
        "wrong_recovery": wrong_recovery,
        "wrong_entropy": wrong_entropy,
        "nokey_recovery": nokey_recovery,
        "nokey_entropy": nokey_entropy,
        "direct_recovery": direct_recovery,
        "direct_entropy": direct_entropy,
        "mutual_info_correct": mutual_info_correct,
        "mutual_info_guess": mutual_info_guess,
        "mutual_info_nokey": mutual_info_nokey,
        "mutual_info_direct": mutual_info_direct
    }

    return metrics

def save_trials(batch_probs: np.ndarray, msgs: list[int], keys: list[int], output_dir: str, filename: str = "trials.csv") -> Path:
    """
    For every (m,k) pair, record the result of decoding with:
    the correct key k, every incorrect key kp, no phase correction, and direct measurement.
    
    Record one row per decoding attempt into CSV.
    Example with n=4, K=3: 
        message,true_key,attempted_key,scenario,recovery,entropy_bits
        0,0,0,correct,0.9999999999999993,9.6e-16
        0,0,1,wrong,0.0269,3.564
        0,0,2,wrong,0.0782,3.628
        0,0,,no_key,0.0139,3.308
        0,0,,direct,0.0625,4.0
        0,1,0,wrong,0.0269,3.564
        ...
    Args:
        batch_probs: 4D array of shape (len(msgs), len(keys), K+2, 2**n) from mk_batch_experiment
        msgs: list of messages tested, in the order they were used
        keys: list of correct keys tested, in the order they were used
        output_dir: directory to save the file
        filename: name of the CSV file
    Returns:
        path: path of the written CSV file
    """

    K = batch_probs.shape[2] - 2 # number of keys

    # get entropies
    trial_entropies = calculate_entropy(batch_probs)

    # get path to save CSV to
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / filename

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        # header row
        writer.writerow(["message", "true_key", "attempted_key", "scenario", "recovery", "entropy_bits"])

        # go through all (message, key) pairs and all key scenarios
        for i, m in enumerate(msgs):
            for j, k in enumerate(keys):
                # attempted key kp
                for kp in range(K + 2):
                    # name the scenario of this attempt
                    # no key is tried for the last two
                    if kp == K:
                        kp_label, scenario = "", "no_key"
                    elif kp == K + 1:
                        kp_label, scenario = "", "direct"
                    else:
                        kp_label = kp
                        scenario = "correct" if kp == k else "wrong"

                    # recovery = probability of reading the message m that was sent
                    # i.e. batch_probs[i, j, kp, m]
                    writer.writerow([m, k, kp_label, scenario,
                                     batch_probs[i, j, kp, m],
                                     trial_entropies[i, j, kp]])

    return path


def average_wrong_keys(trial: np.ndarray, k: int) -> np.ndarray:
    """
    Average the output distributions of all wrong keys for one (message, key) pair.

    Args:
        trial: (K+2, d) probabilities of one pair, as returned by mk_pair_experiment
        k: the correct key
    Returns:
        wrong_dist: vector of d probabilities, one for each basis state
    """

    K = trial.shape[0] - 2  # number of keys
    # rows 0..K-1 of trial are the key attempts, skip the correct one
    wrong_keys = [kp for kp in range(K) if kp != k]

    # average outcome distribution over the K-1 wrong keys
    wrong_dist = trial[wrong_keys].mean(axis=0)
    return wrong_dist

def mutual_info_fixed_decoding(cond_probs: np.ndarray) -> float:
    """
    Calculate the mutual information I(m; mp) between the sent message m
    and the measured output mp, given that Eve uses a fixed decoding strategy.
    The result is in bits.

    NOTE: Conceptually, mutual information I(m; mp) is how much seeing mp reduces the uncertainty about m,
          defined as: I(m; mp) = H(m) - H(m|mp),
          where H(m) is the uncertainty (entropy) about m before measuring and
          H(m|mp) is the conditional uncertainty about m that remains after seeing mp.
          
          Since cond_probs gives P(mp|m) directly, we use the equivalent form
            I(m; mp) = H(mp) - H(mp | m)
          The equivalence due to Bayes rule:
            P(m|mp)/P(m) = P(mp|m)/P(mp)
          and the conditional entropy definition
            H(m | mp) = - sum over m and mp of P(m, mp) * log P(m | mp)

    Args:
        cond_probs: 2D array of shape (M, d) 
                    cond_probs[m, mp] is P(mp | m) for one fixed decoding (specific key kp, no key, direct)

    Returns:
        info: mutual information in bits
    """

    cond_entropy = np.mean(calculate_entropy(cond_probs))
    output_entropy = calculate_entropy(cond_probs.mean(axis=0))
    

    return max(0.0, output_entropy - cond_entropy)  # ensure non-negative

def MAP_fixed_decoding(cond_probs: np.ndarray) -> float:
    """
    Calculate the maximum a posteriori (MAP) probability of correctly guessing the sent
    message m given the measured output mp, given that Eve uses a fixed decoding strategy.
    The MAP probability is calculated as max_m P(m|mp) for each output mp,
    and then marginalizing over all outputs mp.

    Args:
        cond_probs: 2D array of shape (M, d)
                    cond_probs[m, mp] is P(mp | m) for one fixed decoding (specific key kp, no key, direct)

    Returns:
        map_prob: MAP probability marginalized over all outputs mp
    """

    # For each output mp, find the largest P(mp|m) over all messages m.
    # This m also has the largest posterior P(m|mp), since all messages are equally likely.
    map_probs = np.max(cond_probs, axis=0) # max_m P(mp|m) for each mp

    P_m = 1/cond_probs.shape[0] # P(m) = 1/M for all m

    # get sum over mp of P(mp) * max_m P(m|mp)
    # by summing over mp of P(m) * max_m P(mp|m), which by Bayes is equivalent
    MAP = np.sum(map_probs * P_m)
    return MAP

def plot_mk_probabilities(batch_probs: np.ndarray, msg_idx: int, key_idx: int, msgs: list[int], keys: list[int], K: int, output_dir: str, filename: str) -> None:
    """
    Plot the decoding distributions of one selected (message, key) pair as bars:
    correct key, wrong key (averaged), no key and direct measurement.
    Saves the figure as filename in output_dir.
    NOTE: The y-axis is cut above the three small distributions, in reality
          the correct-key bar continues to 1.0.

    Args:
        batch_probs: 4D array of shape (len(msgs), len(keys), K+2, 2**n) 
                     containing the probabilities for each message-key pair.
        msg_idx: index of the sent message in msgs
        key_idx: index of the true key in keys
        msgs: list of possible messages
        keys: list of keys to test
        K: number of keys
        output_dir: directory to save the plot
        filename: name of the saved file
    """

    m = msgs[msg_idx]
    k = keys[key_idx]
    d = batch_probs.shape[-1]
    outcomes = np.arange(d)     # x-axis: output basis states 0..d-1

    # the K+2 decoding attempts for this one (message, key) pair
    trial = batch_probs[msg_idx, key_idx]
    # get distributions for different decoding-key scenarios
    correct_dist = trial[k]
    wrong_dist = average_wrong_keys(trial, k)
    nokey_dist = trial[K]
    direct_dist = trial[K + 1]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # one set of bars per decoding-key scenario, drawn on top of each other
    ax.bar(outcomes, correct_dist, width=0.8, color="#2a78d6", label=f"correct key (k={k})")
    ax.bar(outcomes, wrong_dist, width=0.8, alpha=0.6, color="#eb6834", label=f"wrong key (mean of {K - 1})")
    ax.bar(outcomes, nokey_dist, width=0.8, alpha=0.6, color="#1baf7a", label="no key")
    ax.bar(outcomes, direct_dist, width=0.8, alpha=0.6, color="#52514e", label="direct measurement (1/d)")

    # cut the y-axis just above the three small distributions, otherwise the
    # correct-key bar at 1.0 would make them flat
    y_max = 1.3 * max(wrong_dist.max(), nokey_dist.max(), direct_dist.max())
    ax.set_ylim(0, y_max)

    # axis labels
    ax.set_xlabel("output basis state")
    ax.set_ylabel("probability")
    #light horizontal grid behind the data
    ax.grid(axis="y", color="lightgrey", lw=0.8)
    ax.set_axisbelow(True) # behind
    # no box around data
    ax.spines[["top", "right"]].set_visible(False)

    # legend in one row above the plot, so it wont cover the data
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=4, frameon=False)
    ax.set_title(f"Decoding distributions for message |{m}> locked with key k={k}", loc="left", pad=28)

    # save to file
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / filename, dpi=200)
    plt.close(fig)




def circuit_resources(n: int, K: int, seed: int = 7) -> dict:
    """
    Count total gates of QFT, Dk, and full circuit, and depth of full circuit.
    Returns:
        resources: dict, total gates for keys "QFT", "Dk", "full",
                   and depth of full circuit for key "depth"
    """
    phases = dlock.sample_phases(n, K, seed)

    qft = QuantumCircuit(n)
    qft.append(QFTGate(n), qft.qubits)

    Dk = QuantumCircuit(n)
    Dk.append(dlock.phase_operator(phases, 1), Dk.qubits)

    # message 0, lock and decode with key 0
    full = dlock.lock(0, 0, phases, n)
    full.compose(dlock.decode(0, phases, n), inplace=True)

    resources = {}
    for name, qc in (("QFT", qft), ("Dk", Dk), ("full", full)):
        # transpile into base multi-qubit and single qubit gates
        t = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
        resources[name] = sum(t.count_ops().values())

    resources["depth"] = t.depth()

    return resources
