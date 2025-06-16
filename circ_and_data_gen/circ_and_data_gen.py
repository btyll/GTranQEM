import os

import pickle
import json
import warnings
import numpy as np
from collections import Counter, OrderedDict
from itertools import combinations, chain
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import networkx as nx
import torch
from json import JSONEncoder
from scipy.linalg import expm  # For matrix exponentials

import qiskit
from qiskit import Aer, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, IBMQ
from qiskit.visualization import circuit_drawer, dag_drawer
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.converters import circuit_to_dag
from qiskit.providers.aer.noise import NoiseModel, ReadoutError, pauli_error, depolarizing_error, errors
from qiskit.providers.aer.noise import coherent_unitary_error
import qiskit.quantum_info as qi
from qiskit.providers.fake_provider import FakeWashingtonV2, FakeCairoV2, FakeLima, FakeHanoi, FakeHanoiV2
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, Statevector, state_fidelity
from qiskit.providers.aer import AerSimulator
from qiskit_aer.primitives import Estimator

import circ_parser
from data_gen_func import *




def circ_generator(args):
    total_circ_num = args.total_circ_num
    qubit_num = args.qubit_num
    max_gate_length = args.max_gate_length
    max_trotter_steps = args.max_trotter_steps
    qubit_type = args.qubit_type
    circ_type = args.circ_type
    noise_type = args.noise_type
    shots = args.shots
    depolar_round = args.depolar_round
    exp_type = args.exp_type

    pauli_er = args.pauli_er
    depolar_er_1q = args.dep_1q
    depolar_er_2q = args.dep_2q
    cohe_theta = args.cohe_theta
    readout_er = args.readout_er

    print("Simulation Parameters:")
    print("----------------------")
    print(f"Total circuits: {total_circ_num}")
    print(f"Number of qubits: {qubit_num}")
    print(f"Max gate length: {max_gate_length}")
    print(f"Max Trotter steps: {max_trotter_steps}")
    print(f"Qubit type: {qubit_type}")
    print(f"Circuit type: {circ_type}")
    print(f"Noise type: {noise_type}")
    print(f"Exp type: {exp_type}")
    print(f"Shots: {shots}")
    print(f"Rounds for adding depolar: {depolar_round}")
    print("\nNoise Error Rates:")
    print("------------------")
    print(f"Pauli error rate: {pauli_er}")
    print(f"1-qubit depolarizing error rate: {depolar_er_1q}")
    print(f"2-qubit depolarizing error rate: {depolar_er_2q}")
    print(f"Coherent error angle (radians): {cohe_theta}")
    print(f"Readout error rates: {readout_er}")

    base_dir = f'/tmp/{exp_type}/{qubit_type}_{qubit_num}q/{circ_type}'
    exp_dir = f'{base_dir}/{noise_type}'
    os.makedirs(exp_dir, exist_ok=True)

    if os.path.exists(f'{base_dir}/all_circ_dict.pkl'):
        all_circ_dict = pickle.load(open(f'{base_dir}/all_circ_dict.pkl', 'rb'))
        print(f'{len(all_circ_dict.keys())} circ dict existed, load')
    else:
        assert False, 'should not use this file for cirq generation'
    
    if qubit_num == 50:
        noise_model = None
        return all_circ_dict, noise_model, base_dir, exp_dir

    if os.path.exists(f'{exp_dir}/noise_model.pkl'):
        print('noise model existed, load')
        noise_model = pickle.load(open(f'{exp_dir}/noise_model.pkl', 'rb'))
    else:
        assert False, 'should not use this file for noise model generation'
    return all_circ_dict, noise_model, base_dir, exp_dir


if __name__ == "__main__":
    allowed_gates = ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 'cx', 'cz', 'swap', 'rzz']
    double_q_gates = ['cx', 'cz', 'swap', 'rzz']
    single_q_gates = [x for x in allowed_gates if x not in double_q_gates]

    args = circ_parser.parse_args()
    all_circ_dict, noise_model, base_dir, exp_dir = circ_generator(args)
    
    len_of_cir = all_circ_dict[args.chosen_con_cir]['circ'].size()
    
    qc = all_circ_dict[args.chosen_con_cir]['circ'].copy()

    ideal_noisy_ev_gen(qc, args.chosen_con_cir, noise_model, args.shots, exp_dir)
    
    get_edge_feat_evs(exp_dir, args.chosen_con_cir, qc, args.qubit_num, noise_model, args)
    cirq_graph = circuit_to_graph(all_circ_dict[args.chosen_con_cir]['circ'])
    adj_mat = get_adjacency_matrix(cirq_graph)
    assert adj_mat.shape[0] == adj_mat.shape[1] == len_of_cir + 2 * args.qubit_num + 1
    floyd_warshall(adj_mat, base_dir, args.chosen_con_cir)
    