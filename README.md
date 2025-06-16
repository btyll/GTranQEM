The official implementation for ICLR25 paper "Beyond Circuit Connections: A Non-Message Passing Graph Transformer Approach Towards Quantum Error Mitigation"

## Dependencies
- python == 3.10.8
- torch == 2.4.0
- torch-geometric == 2.5.3
- qiskit == 0.46.2
- qiskit-aer == 0.13.3


## Dataset preparation
Download a sample dataset from [all-6-trotter-incoherent](https://drive.google.com/file/d/1QV7agcKjjJly450j53suHl8SWYWPvTpZ/view?usp=sharing) or generate your own dataset via code in the *circ_and_data_gen* folder

## How to run the code

For GTranQEM, please run

```bash
$ pip install -r requirement.txt
$ python train.py ev_6bit_trotter_all
```
You can modify the model hyperparameters in *exp/ev_6bit_trotter_all/config.yaml* for the sample accordingly.

