"""Validation tools."""
import torch
import torch.nn as nn
import pytorch_lightning as pl

from datetime import datetime

import os
import json


import optuna

from typing import List, Tuple, Optional, Union, Dict, Any

from exp.exp import LightningModel 


def train_lightning_model(args, test=False) -> Tuple[
    nn.Module,
    pl.loggers.TensorBoardLogger,
    str,
]:
    """Prepare datasets and initialize, train, and test model.

    :param df: initial dataframe
    :param train_index: indices of training elements
    :param test_index: indices of testing elements
    :param well_column: name of column with names of wells
    :param slice_len: the length of well-interval
    :param save_dir: directory for saving scaler
    :param results_len_train: the number of well-intervals for training
    :param results_len_test: the number of well-intervals for testing
    :param model: model for training and testing
    :param model_type: type of model
    :param batch_size: batch size
    :param log_dir: directory for saving logs
    :param epochs: the number of epochs for model training
    :param gpu: if is not 0 or None use GPU (else CPU)
    :return: tuple of
             - training and validation datasets
             - trained model
             - logger (for further results plotting)
             - experiment name (for necessary information saving)
    """

    model = LightningModel(args)

    current_time = datetime.now().strftime("%m%d%Y_%H:%M:%S")
    experiment_name = args.model + "_" + current_time.replace(":", "_")

    logger = pl.loggers.TensorBoardLogger(save_dir=args.log_dir, name=experiment_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{args.log_dir}/{experiment_name}",
        filename="{epoch:02d}-{val_loss:.3f}",
        mode="min",
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode="min",
    )

    accelerator = 'gpu' if args.use_gpu else 'cpu'
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices='auto',
        accelerator=accelerator,
        benchmark=True,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model)
    if test:
        trainer.test(model)
    
    callback_metrics = trainer.callback_metrics
    
    return (
        model,
        callback_metrics,
        logger,
        experiment_name,
    )


def optuna_hpo_and_best_model_evaluation(
    n_trials: int,
    specific_params: Dict[str, Any],
    args,
) -> Tuple[nn.Module, Dict[str, float], List[float]]:
    """Select hyperparameters with Optuna and train and evaluate model with the best set of hyperparameters.

    :param model_type: type of model
    :param device: calculation device ('cpu' or 'cuda')
    :param gpu: indicate used gpu index
    :param n_trials: the number of optuna iterations
    :param fixed_params: parameters that would not be changed
    :param default_params: default values of hyperparameters
    :param specific_params: hyperparameter space
    :param data_kwargs: arguments for data processing
    :return: tuple of trained model, metrics, and ROC AUC scores from all Optuna trials
    """

    print("model_type:", args.model)

    def objective(trial):
        def get_specific_params(specific_params):
            ans = dict()
            for k, (suggest_type, suggest_param) in specific_params.items():
                if suggest_type == "cat":
                    ans[k] = trial.suggest_categorical(k, suggest_param)
                elif suggest_type == "int":
                    ans[k] = trial.suggest_int(k, *suggest_param)
                elif suggest_type == "float":
                    print(k, suggest_param)
                    ans[k] = trial.suggest_float(k, *suggest_param)
            return ans

        trial_specific_params = get_specific_params(specific_params)

        args.__dict__.update(**trial_specific_params)
        
        
        (
            lightning_model,
            callback_metrics,
            logger,
            experiment_name,
        ) = train_lightning_model(args)

        metric = callback_metrics["val_mse"]

        torch.save(
            lightning_model.model.state_dict(),
            f"{args.save_dir}{args.model}_{experiment_name}_{metric}.pth",
        )

        return metric
    
    study = optuna.create_study(direction="minimize", storage="sqlite:///db.sqlite3")
    study.enqueue_trial(args.__dict__)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    with open(
        os.path.join(args.save_dir, "best_params_{}.json".format(args.model)),
        "w",
    ) as f:
        json.dump(best_params, f)
    print("best_params:", best_params)

    mse_all_trials = [trial.value for trial in study.trials]

    args.__dict__.update(best_params)

    (
        lightning_model,
        callback_metrics,
        logger,
        experiment_name,
    ) = train_lightning_model(args, test=True)


    test_metrics = dict(mse=callback_metrics["test_mse"].item(), 
                        mae=callback_metrics["test_mae"].item(), 
                        rmse=callback_metrics["test_rmse"].item(), 
                        mape=callback_metrics["test_mape"].item(),
                        mspe=callback_metrics["test_mspe"].item())
    
    res_model = lightning_model.model

    torch.save(
        res_model.state_dict(),
        os.path.join(args.save_dir, "best_{}.pth".format(args.model)),
    )

    print("\nall_results:", test_metrics)
    print("*" * 100)

    return res_model, test_metrics, mse_all_trials
