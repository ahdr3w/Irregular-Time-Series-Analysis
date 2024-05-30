import os
import warnings
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import random


import logging
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


from tqdm import tqdm

warnings.filterwarnings("ignore")

from copy import deepcopy
from exp import utils_cv
from utils.print_args import print_args
torch.set_float32_matmul_precision('high')

def fix_seeds(seed_value: int = 42, device: str = "cpu") -> None:
    """Source 'https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5'.

    :param seed_value: random state value
    :param device: device for which seeds would be fixed
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != "cpu":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

save_dir = "./save_transformer_cab/"
log_dir = "./log_transformer_cab/"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='imputation',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='TransformerCAB',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--device_id', type=int, default='1', help='device id')


# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# metrics (dtw)
parser.add_argument('--use_dtw', type=bool, default=False, 
                    help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

# Augmentation
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

parser.add_argument('--corr_beta', type=float, default='0.0',help='corr_beta')
parser.add_argument('--corr_lambda', type=float, default='0.0',help='corr_lambda')
parser.add_argument('--corr_tau', type=float, default='1.0',help='corr_tau')
parser.add_argument('--n_corr_heads', type=int, default='8', help='num of corr heads')

# parser.add_argument('--n_warmup', type=int, default=5, help='num of warmup steps')
# parser.add_argument('--sch_gamma', type=float, default='0.94',help='scheduler gamma')
# parser.add_argument('--sch_start_factor', type=float, default='0.21',help='scheduler start factor')


parser.add_argument('--save_dir', type=str, default='./save_transformer_cab/',help='save dir')
parser.add_argument('--log_dir', type=str, default='./log_transformer_cab/',help='log dir')


parser.add_argument('-f', type=str, default='None')

args = parser.parse_args()

from exp.exp import LightningModel
import pytorch_lightning as pl
import json

accelerator = 'gpu' if args.use_gpu else 'cpu'
trainer = pl.Trainer(
    max_epochs=args.epochs,
    devices=[args.device_id],
    accelerator=accelerator,
    check_val_every_n_epoch=1,
    enable_progress_bar=False
    
)

def test(model_name, ds_name, args):
    metrics = dict()
    miss_rate = [0.125, 0.25, 0.375, 0.5]


    
    
    args.model = model_name
    args.data = ds_name
    args.data_path = ds_name + '.csv'
    
    lightning_model = LightningModel(args)
    
    for i in tqdm(range(len(miss_rate)), desc='[0.125, 0.25, 0.375, 0.5]', leave=False):
        metrics[f"on_train_miss_{miss_rate[i]}"] = dict()

        maes = [[]]
        mses = [[]]
        for seed in tqdm([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], desc='seeds', leave=False):
            postfix = f'{int(100*miss_rate[i])}_{int(10*(100*miss_rate[i] - int(100*miss_rate[i])))}_seed_{seed}'
            lightning_model.load_model(f"model_pt/{ds_name}_{model_name}_params_miss_{postfix}.pt")
            
            for j in tqdm(range(len(miss_rate)), desc='Test missing rate', leave=False):
                args.mask_ratio = miss_rate[j]
                lightning_model.update_args(args)
                
                mae_avg = 0
                mse_avg = 0
                for _ in tqdm(range(3), leave=False):
                    trainer.test(lightning_model, verbose=False)
                    mae, mse = trainer.callback_metrics["test_mae"], trainer.callback_metrics["test_mse"]
                    mae_avg += mae
                    mse_avg += mse
                mae_avg /= 3
                mse_avg /= 3
                maes[-1].append(mae_avg)
                mses[-1].append(mse_avg)
            maes.append([])
            mses.append([])
            
        maes = np.array(maes[:-1])
        mses = np.array(mses[:-1])
        mean_maes = maes.mean(0)
        mean_mses = mses.mean(0)
        std_maes = maes.std(0)
        std_mses = mses.std(0)
                
        for j in range(len(miss_rate)):        
            metrics[f"on_train_miss_{miss_rate[i]}"][f"on_test_miss_{miss_rate[j]}"] = [('MAE', [mean_maes[j], std_maes[j]]), ('MSE', [mean_mses[j], std_mses[j]])]
            
    return metrics


def compile_table(model_name, ds_name, metrics):
    
    mse = [[f"{ds_name}_{model_name}_MSE","on_test_miss_0.125","on_test_miss_0.25","on_test_miss_0.375","on_test_miss_0.5"]]
    mae = [[f"{ds_name}_{model_name}_MAE","on_test_miss_0.125","on_test_miss_0.25","on_test_miss_0.375","on_test_miss_0.5"]]
    for k, v in metrics.items():
        mse.append([f"{k}"])
        mae.append([f"{k}"])
        for kk, vv in v.items():
            mae[-1].append(f"{vv[0][1][0]:.3f} ± {vv[0][1][1]:.3f}")
            mse[-1].append(f"{vv[1][1][0]:.3f} ± {vv[1][1][1]:.3f}")
    
    mae = pd.DataFrame(mae)
    mse = pd.DataFrame(mse)
    
    mse.to_csv(f"stats/{ds_name}_{model_name}_mse_on_irregular_data.csv")
    mae.to_csv(f"stats/{ds_name}_{model_name}_mae_on_irregular_data.csv")


def plot_model_metrics_with_std(means, stds, model_name):
    plt.title(model_name)
    x = [12.5, 25.0, 37.5, 50.0]
    for mean, std, label in zip(means, stds, [f'Train Miss Level {p:.1%}' for p in [0.125, 0.25, 0.375, 0.5]]):
        plt.errorbar(x, mean, yerr=std, label=label, fmt='-o', capsize=5)
    plt.xlabel('Test Missing Data Level, %')
    plt.ylabel('Metric Score')
    plt.legend()

def plot_dataset_models_with_std(all_data, all_stds, model_names, dataset_name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(dataset_name)
    
    for ax, data, stds, name in zip(axes, all_data, all_stds, model_names):
        plt.sca(ax)
        plot_model_metrics_with_std(data, stds, name)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.savefig(f'png/{dataset_name}.png', bbox_inches='tight')
    plt.show()

def data_collection(args):

    data_mae = []
    data_mse = []
    
    for ds_name in tqdm(["ETTh1", "ETTh2", "ETTm1", "ETTm2"], desc='["ETTh1", "ETTh2", "ETTm1", "ETTm2"]', leave=False):
        
        mean_maes = []
        std_maes  = []
        mean_mses = []
        std_mses  = []
        
        for model_name in tqdm(["Transformer", "Informer", "TransformerCAB"], desc='["Transformer", "Informer", "TransformerCAB"]', leave=False):
            
            metrics = test(model_name, ds_name, args)
            compile_table(model_name, ds_name, metrics)
            
            mean_mae = []
            std_mae  = []
            mean_mse = []
            std_mse  = []
            
            for k, v in metrics.items():
                mean_mae.append([])
                std_mae.append([])
                mean_mse.append([])
                std_mse.append([])
                
                for kk, vv in v.items():
                    mean_mae[-1].append(vv[0][1][0])
                    std_mae[-1].append(vv[0][1][1])
                    mean_mse[-1].append(vv[1][1][0])
                    std_mse[-1].append(vv[1][1][1])
                    
            mean_mae = np.array(mean_mae)
            std_mae = np.array(std_mae)
            mean_mse = np.array(mean_mse)
            std_mse = np.array(std_mse)

            mean_maes.append(mean_mae)
            std_maes.append(std_mae)
            mean_mses.append(mean_mse)
            std_mses.append(std_mse)

        mean_maes = np.stack(mean_maes)
        mean_maes = np.stack(std_maes)
        mean_mses = np.stack(mean_mses)
        std_mses = np.stack(std_mses)

        data_mae.append((ds_name+'_mae', mean_maes, std_maes))
        data_mse.append((ds_name+'_mse', mean_mses, std_mses))

    return data_mae, data_mse

data_mae, data_mse = data_collection(args)

model_names = ["Transformer", "Informer", "TransformerCAB"]
for dataset_name, means, stds in data_mae:
    plot_dataset_models_with_std(means, stds, model_names, dataset_name)

for dataset_name, means, stds in data_mse:
    plot_dataset_models_with_std(means, stds, model_names, dataset_name)


np.save("stats/mae_metric_log", data_mae)
np.save("stats/mse_metric_log", data_mse)