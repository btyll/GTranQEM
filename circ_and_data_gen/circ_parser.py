import argparse
import numpy as np

def str2bool(v):
    """Convert a string to a boolean value for argument parsing."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Parse command-line arguments and return the parsed parameters."""
    parser = argparse.ArgumentParser(description='Parse simulation parameters.')

    # Simulation parameters
    parser.add_argument('--total_circ_num', type=int, default=400,
                        help='Total number of circuits.')
    parser.add_argument('--qubit_num', type=int, default=2,
                        help='Number of qubits.')
    parser.add_argument('--max_gate_length', type=int, default=150,
                        help='Maximum gate length.')
    parser.add_argument('--max_trotter_steps', type=int, default=20,
                        help='Maximum Trotter steps.')
    parser.add_argument('--qubit_type', type=str, default='all', choices=['max', 'all'],
                        help='Type of qubits to use ("max" N qubit or "all" N qubit).')
    parser.add_argument('--circ_type', type=str, default='trotter', choices=['random', 'trotter'],
                        help='Type of circuit ("random" or "trotter").')
    parser.add_argument('--noise_type', type=str, default='all_noise',
                        choices=['incoherent', 'inco_cohe', 'all_noise', 'provider', 'coherent'],
                        help='Type of noise model to use.')
    
    parser.add_argument('--exp_type', type=str, default='standard',
                    choices=['standard', 'mirror', 'amplify'],
                    help='Type of noise model to use.')
    parser.add_argument('--depolar_round', type=int, default=6,
                        help='Number of random rounds for adding depolar error')
    parser.add_argument('--shots', type=int, default=1000,
                        help='Number of shots for the simulation.')

    # Noise error rates
    parser.add_argument('--pauli_er', type=float, default=0.000437194,
                        help='Pauli error rate.')
    parser.add_argument('--dep_1q', type=float, default=0.0032,
                        help='1-qubit depolarizing error rate.')
    parser.add_argument('--dep_2q', type=float, default=0.0124,
                        help='2-qubit depolarizing error rate.')
    parser.add_argument('--cohe_theta', type=float, default=0.2617993878,
                        help='Coherent error angle in radians (default is pi/12).')
    parser.add_argument('--readout_er', type=float, nargs=2, default=[0.023, 0.011],
                        help='Readout error rates as two floats.')

    # for large qubit training
    parser.add_argument('--chosen_con_cir', type=int, default=0,
                    help='The circ_idx of the chosen cirq to generate evs')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Access the variables
    total_circ_num = args.total_circ_num
    qubit_num = args.qubit_num
    max_gate_length = args.max_gate_length
    is_circ_gen = args.is_circ_gen
    max_trotter_steps = args.max_trotter_steps
    qubit_type = args.qubit_type
    circ_type = args.circ_type
    noise_type = args.noise_type
    shots = args.shots

    pauli_er = args.pauli_er
    depolar_er_1q = args.depolar_er_1q
    depolar_er_2q = args.depolar_er_2q
    cohe_theta = args.cohe_theta
    readout_er = args.readout_er

    # Now you can use these variables in your code
    print("Simulation Parameters:")
    print("----------------------")
    print(f"Total circuits: {total_circ_num}")
    print(f"Number of qubits: {qubit_num}")
    print(f"Max gate length: {max_gate_length}")
    print(f"Is circuit generation enabled: {is_circ_gen}")
    print(f"Max Trotter steps: {max_trotter_steps}")
    print(f"Qubit type: {qubit_type}")
    print(f"Circuit type: {circ_type}")
    print(f"Noise type: {noise_type}")
    print(f"Shots: {shots}")
    print("\nNoise Error Rates:")
    print("------------------")
    print(f"Pauli error rate: {pauli_er}")
    print(f"1-qubit depolarizing error rate: {depolar_er_1q}")
    print(f"2-qubit depolarizing error rate: {depolar_er_2q}")
    print(f"Coherent error angle (radians): {cohe_theta}")
    print(f"Readout error rates: {readout_er}")
