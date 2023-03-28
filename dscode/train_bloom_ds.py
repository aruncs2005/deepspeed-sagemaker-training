import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import collections
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import torch
import deepspeed
from typing import List, Tuple

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoModelForCausalLM,

    SchedulerType,
    get_scheduler,
    BloomConfig,
    set_seed
)

import argparse



class OpenwebtextPretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, input_paths: List[str], max_sequence_length=None, use_last_file_only=False):
        self.input_paths = input_paths
        self.max_sequence_length = max_sequence_length
        self.use_last_file_only = use_last_file_only

        self.__read_examples(self.input_paths)

    def __read_examples(self, paths: List[str]):

        self.input_data = []
   
        if self.use_last_file_only:
            with open (paths[-1], "r") as f:
                self.input_data = [ln for ln in f]
        else:
            for path in paths:
                with open (path, "r") as f:
                    self.input_data.extend([ln for ln in f])

        # print(f'__Finished building pretraining dataset with {self.iids.shape[0]} rows__')

    def __len__(self) -> int:
        return len(self.input_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = json.loads(self.input_data[index])
        iids = torch.tensor(obj["input_ids"], dtype=torch.long)
        attns = torch.tensor(obj["attention_mask"], dtype=torch.long)
        self.actual_sequence_length = len(obj["input_ids"])

        if self.actual_sequence_length > self.max_sequence_length:
            s_idx = np.random.randint(0, self.actual_sequence_length - self.max_sequence_length)
            e_idx = s_idx + self.max_sequence_length
            iids = iids[s_idx:e_idx]
            attns = attns[s_idx:e_idx]
        return iids, attns



def read_args():

    parser = argparse.ArgumentParser(description="Train a transformers model from scratch on causal language modeling")

    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")

    parser.add_argument("--max_seq_length", type=int, default=512, help="Sequence length.")

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default= 16,
        help="batch size per dp rank, for tensor parallelism degree 8 with pipeline parallel degree 1 this means 8*this batch size per node",
    )
    parser.add_argument("--val_batch_size", type=int, default=16)

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
 
    parser.add_argument(
        "--validation_batches",
        type=int,
        default=10,
        help="number of batches to estimate validation loss",
    )

    parser.add_argument("--training-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])


    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])

    parser.add_argument("--max_eval_steps", type=int, default=100)
    parser.add_argument("--store_final_model", type=bool, default=False)

    parser.add_argument("--distributed_backend",type=str, default="nccl")

   

    #include deepspeed configuration 
    parser = deepspeed.add_config_arguments(parser)


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    num_nodes = 0
    sm_config_path = '/opt/ml/input/config/resourceconfig.json'
    if os.path.exists(sm_config_path):
        with open(sm_config_path) as file:
            cluster_config = json.load(file)

        hosts = cluster_config['hosts']
        print("*****printing list of hosts **********")
        print(hosts)
        num_nodes = len(hosts)
        print("Total number of nodes in the training cluster - {}".format(num_nodes))
       
 
    args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
    args.local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))
    
    args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    args.world_size = num_nodes*args.local_size

    print("local rank : {}, global rank : {} , local size : {},  world size : {}".format(args.local_rank,args.rank, args.local_size, args.world_size))
    print("logging all the arguments captured")
    print(args)

    return args



def compute_num_params(model):
    num_params = 0
    seen = set()
    for p in model.parameters():
        if p not in seen:
            seen.add(p)
            if hasattr(p, "ds_shape"):
                num_params += np.prod(p.ds_shape)
            else:
                num_params += np.prod(p.size())

    return num_params

def train_step(model, optimizer, input_ids, attention_mask, args):

    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)["loss"]
  
    model.backward(loss)
  
    return loss



def test_step(model, input_ids, attention_mask):
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)["loss"]
    return loss


def eval_model(model, dataloader, local_rank,global_rank):
    print("Running evaluation loop")
    model.eval()
    losses = []
    for step, batch in enumerate(dataloader):
        if global_rank == 0:
            print("inside the eval loop step no {}".format(step))
        with torch.no_grad():
            input_ids, attention_mask = batch
            loss = test_step(model, input_ids.to(local_rank), attention_mask.to(local_rank))
            
            losses.append(loss)
        if step > 100:
            break
    try:
        eval_loss = torch.mean(torch.tensor(losses,dtype=float))
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return eval_loss, perplexity



def main():
    args = read_args()


    #if args.deepspeed:
        #deepspeed.init_distributed(dist_backend=args.distributed_backend)
    
    model_config = BloomConfig(vocab_size = 250880,
                                n_embed = 2560,
                                initializer_range =  0.02,
                                layer_norm_epsilon =  1e-05,
                                n_layer = 30,
                                num_attention_heads = 32,
                                offset_alibi = 100,
                                pretraining_tp = 4,
                                seq_length = 2048,
                                use_cache = False,
                                bos_token_id = 1,
                                eos_token_id = 2,
                                pad_token_id = 3,
                                unk_token_id = 0,
                                skip_bias_add = True,
                                skip_bias_add_qkv = False,
                                attention_softmax_in_fp32 = True,
                                apply_residual_connection_post_layernorm = False,
                                bias_dropout_fusion = True,
                                hidden_dropout = 0.0,
                                attention_dropout = 0.0,
                                slow_but_exact = False)


    set_seed(args.seed)
    # we are loading the model to prevent DeepSpeed from erroring out during zero init.
    model = AutoModelForCausalLM.from_config(model_config)
    
    with deepspeed.zero.Init(remote_device="cpu",pin_memory="True"):
        model = AutoModelForCausalLM.from_config(model_config)
        num_params = compute_num_params(model)

    # load the data into data loader for both training and validation, here we will load the tokenized data from previous step to avoid re computations for repeated training.
    file_extension = "json"
    train_paths = sorted(
            [
                os.path.join(args.training_dir, p)
                for p in os.listdir(args.training_dir)
                if p.endswith(file_extension)
            ]
        )
    print("Number of files in the training directory {}".format(len(train_paths)))

    train_ds = OpenwebtextPretrainingDataset(
                    input_paths=train_paths, max_sequence_length=args.max_seq_length,use_last_file_only=False
                )



    val_paths = sorted(
                [
                    os.path.join(args.test_dir, p)
                    for p in os.listdir(args.test_dir)
                    if p.endswith(file_extension)
                ]
            )

    print("Number of files in the validation directory {}".format(len(val_paths)))
 
    eval_ds = OpenwebtextPretrainingDataset(
                    input_paths=val_paths, max_sequence_length=args.max_seq_length,use_last_file_only=False
                )

    eval_dataloader = torch.utils.data.DataLoader(
                eval_ds,
                batch_size=args.val_batch_size,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

 

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    
    
    ds_config = json.load(open('dsconfig.json'))

    model, optimizer,train_dataloader, lr_scheduler = deepspeed.initialize(
            model=model,
            #optimizer=optimizer,
            args=args,
            training_data=train_ds,
            #lr_scheduler=lr_scheduler,
            config=ds_config)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    if args.rank == 0:
        print("Total number of parameter is {}".format(num_params))
        print("max training steps for the job {}".format(args.max_train_steps))

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda")

    total_steps = 0
    start = time.time()
    starting_epoch = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if total_steps > args.max_train_steps:
                break
            if args.rank == 0:
                print("Training step {} of total {}".format(step,total_steps))
            step_start = time.time()
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device),attention_mask.to(device)
            loss_mb = train_step(model, optimizer, input_ids, attention_mask, args)
            loss = loss_mb
            optimizer.step()

            total_steps += 1
            time_elapsed = time.time() - start
            step_time = time.time() - step_start
            sample_processed = input_ids.shape[0] * args.world_size
            throughput = sample_processed / step_time
            tokens_per_gpu = input_ids.shape[0] * input_ids.shape[1]

            tflops_per_gpu = 8 * num_params * tokens_per_gpu / step_time / 1e12
            
            if args.rank == 0 and total_steps % 10 == 0: 
                print(
                    f"({int(time_elapsed)}s elapsed train time) Batch: {total_steps}, Loss: {loss.item()}, Speed: {throughput} samples/sec, TFLOPS/GPU: {tflops_per_gpu}"
                )
                
           

        val_loss, val_ppl = eval_model(
                    model, eval_dataloader, args.local_rank,args.rank
                )

       

        print(f"epoch {epoch}: perplexity: {val_ppl} eval_loss: {val_loss}")
    
 
    if args.rank == 0:
        print("training successfully completed!!!!!")
    


if __name__ == "__main__":
    main()