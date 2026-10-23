import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate, DiagonalGate
from qiskit.circuit import Gate


def message(m: int, n: int) -> QuantumCircuit:
    """Represent a message m as an n-qubit basis state ∣m⟩.
    """
    msg_reg = QuantumRegister(n, "msg")
    qc = QuantumCircuit(msg_reg, name=f"|{m}>")

    # get m as bitstring in little-endian order 
    # matches qiskit: q0 is LSB
    bits = format(m, f"0{n}b")[::-1]
    for i, bit in enumerate(bits): # loop through each bit position in bitstring
        if bit == "1":
            qc.x(msg_reg[i]) # flip i:th bit in register to 1 
    return qc


def sample_phases(n: int, K: int, seed: int = 7, binary: bool = False) -> np.ndarray:
    """Sample d*K phase angles Thetas[k,w], d = 2**n.
        Each angle sampled uniformly from [0,2pi), or from {0, pi} if binary is True.
        Returns:
            Thetas, shape (K, d)
    """
    d = 2**n
    rng = np.random.default_rng(seed)
    if binary:
        # each angle is 0 or pi, so each diagonal entry of Dk is +1 or -1
        Thetas = np.pi * rng.integers(0, 2, size=(K, d))
    else:
        Thetas = rng.uniform(0.0, 2*np.pi, size=(K, d))
    return Thetas


def phase_operator(phases: np.ndarray, k: int) -> Gate:
    """Construct the phase operator Dk = diag(...)
    Args:
        phases: all (K, d) sampled phases in codebook
        k: key number
    Returns:
        Dk: diagonal phase operator for key k
    """
    Dk = DiagonalGate(np.exp(1j*phases[k,:]))
    # DiagonalGate gives too long name when drawing diagram
    # SOLUTION: wrap Dk in QuantumCircuit with different name and convert to gate
    d = phases.shape[1] # d = 2**n
    n = round(np.log2(d))
    qc = QuantumCircuit(n, name=f"D{k}")
    qc.append(Dk, qc.qubits)
    return qc.to_gate()


def lock(m: int, k: int, phases: np.ndarray, n: int) -> QuantumCircuit:
    """Prepare the locked state |ψ_mk>."""

    msg_reg = QuantumRegister(n, "msg")
    qc = QuantumCircuit(msg_reg, name="lock")

    # 1. Represent a message m as an n-qubit basis state ∣m⟩.
    qc.append(message(m, n).to_gate(), msg_reg) # to_gate so it draws nicely
    # 2. Apply a quantum Fourier transform F to the message.
    qc.append(QFTGate(n), msg_reg) 
    # 3. Let the key k select a diagonal phase operator Dk
    Dk = phase_operator(phases, k)
    # 4. Prepare the locked state |ψ_mk> = Dk*F*|m> 
    qc.append(Dk, msg_reg)

    return qc


def decode(k: int | None, phases: np.ndarray, n: int) -> QuantumCircuit:
    """Circuit to decode with the key k, or no key (None), by applying F†*Dk†.
    """
    msg_reg = QuantumRegister(n, "msg")
    qc = QuantumCircuit(msg_reg, name="decode")

    # Apply Dk† if using a key
    if k is not None:
        Dk = phase_operator(phases, k)
        qc.append(Dk.inverse(), msg_reg)

    # Apply F†
    qc.append(QFTGate(n).inverse(), msg_reg)

    return qc



def draw_example(m: int, k: int, kp: int | None, phases: np.ndarray, n: int, path: str) -> None:
    """Draw the complete message -> lock -> decode circuit and save it as an image.
    Args:
        m: message sent
        k: key used to lock
        kp: key used to decode, or None for no key
        phases: all (K, d) sampled phases in codebook
        n: number of qubits
        path: file to save the diagram image to
    """

    # lock with key k, then decode with key kp
    qc = lock(m, k, phases, n)
    qc.compose(decode(kp, phases, n), inplace=True)

    # draw circuit diagram and save 
    fig = qc.draw("mpl", fold=-1)
    fig.savefig(path)
    plt.close(fig)