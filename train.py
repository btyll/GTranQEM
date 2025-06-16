
import argparse
import random
import sys
import time

import core.datasets.builder as builder
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from core.datasets.trainer import trainer
from torch_geometric.loader import DataLoader
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from data_loader import load_ev_dataset
import os, json



def main() -> None:
    seed = 233
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    configs.evalmode = False

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", metavar="FILE", help="config file") 
    parser.add_argument("--load", action="store_true", help="config file")
    parser.add_argument("--special_node_with_grad", action="store_true", help="config file")
    parser.add_argument("--train_size", default=0, type=int, help="config file")

    args, opts = parser.parse_known_args()
    print("Args", args)

    configs.load(f"exp/{args.exp_name}/config.yaml", recursive=True)
    configs.update(opts)
    configs.exp_name = args.exp_name
    if configs.device == "gpu":
        device = torch.device("cuda")
    elif configs.device == "cpu":
        device = torch.device("cpu")

    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)
    args.run_dir = "/tmp"

    logger.info(" ".join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + "\n" + f"{configs}")

    model = builder.make_model(args)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Size: {total_params}")
    dataflow = {}

    mydataset, train_dataset_list, valid_dataset_list, test_dataset_list, num_features = load_ev_dataset(configs.data_path, configs.num_qubits, device, test_type=configs.exp_type, args=args, train_size=configs.train_size, valid_size=configs.valid_size, test_size=configs.test_size)
    dataflow["train"] = DataLoader(train_dataset_list, batch_size=configs.batch_size)
    dataflow["valid"] = DataLoader(valid_dataset_list, batch_size=configs.batch_size)
    dataflow["test"] = DataLoader(test_dataset_list, batch_size=configs.batch_size)

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    print(criterion)
    my_trainer = trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataflow,
    )

    training_time = -1
    test_time = -1
    if not args.load:
        training_start_time = time.time()
        my_trainer.train()
        training_end_time = time.time()
        test_start_time = time.time()
        my_trainer.test()
        test_end_time = time.time()
        training_time = training_end_time - training_start_time
        test_time = test_end_time - test_start_time
        print(f"Training time: {training_end_time - training_start_time:.2f}s")
        print(f"Testing time: {test_end_time - test_start_time:.2f}s")
        my_trainer.saveall()
    
    test_pred, test_y, test_cir_id = my_trainer.training_data['test_pred'], my_trainer.training_data['test_y'], my_trainer.training_data['test_cir_id']
    save_dict = {}
    for pred, y, cir_id in zip(test_pred, test_y, test_cir_id):
        save_dict[cir_id] = {
            'ours': pred,
            'gt': y,
        }
    os.makedirs(f'{configs.data_path}/results', exist_ok=True)
    os.makedirs(f'{configs.data_path}/results/time_profile', exist_ok=True)

    if args.train_size > 0:
        json.dump(save_dict, open(f"{configs.data_path}/results/ours_ev_dict_{args.train_size}.json", "w"), indent=4)
    else:   
        json.dump(save_dict, open(f"{configs.data_path}/results/ours_ev_dict.json", "w"), indent=4)
    assert training_time != -1 and test_time != -1
    print(f"Training time: {training_time:.2f}s")
    print(f"Testing time: {test_time:.2f}s")
    profile_path = f"{configs.data_path}/results/time_profile/ours_time_profile.json"
    json.dump({
        'training_time': training_time,
        'test_time': test_time,
    }, open(profile_path, "w"), indent=4)
   

    my_trainer.loadall()
    my_trainer.curve()
    my_trainer.scatter()


if __name__ == "__main__":
    main()
