from typing import Dict, List, Any

import numpy as np

import pytorch_lightning as pl

import torch
from torch import nn
from torch.utils.data import DataLoader


from models import TransformerCAB, Transformer, Informer
from utils.metrics import metric

from data_provider.data_factory import data_provider


MODEL_DICT = {
    'TransformerCAB':TransformerCAB,
    'Transformer':Transformer,
    'Informer':Informer,
}


class LightningModel(pl.LightningModule):
    """PyTorch Lightning wrapper for convenient training models."""

    def __init__(self, args) -> None:
        """Initialize PyTorch Wrapper.

        :param model_type: model type in format <siamese/triplet>_<reguformer/transformer/performer>
        :param train_data: Dataset with training data
        :param test_data: Dataset with test data
        :param batch_size: batch size for DataLoader
        :param learning_rate: hyperparameter that controls how much to change the model in response to
            the estimated error each time the model weights are updated.
        """
        super(LightningModel, self).__init__()

        self.args = args

        self.model = self._build_model()

        self.val_step_outputs = []
        self.test_step_outputs = []        

        self.train_loss_log = []
        self.val_loss_log = []
        self.train_loss = []
        self.val_loss = []
        
        self.loss_function = nn.MSELoss()

    def _build_model(self):
        model = MODEL_DICT[self.args.model].Model(self.args).float()
        
        return model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get model's output.

        :param inputs: input data for model
        :return: base model's data
        """
        return self.model(inputs)
    

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float()
        batch_x_mark = batch_x_mark.float()

        # random mask
        B, T, N = batch_x.shape
        mask = torch.rand((B, T, N)).to(self.device)
        mask[mask <= self.args.mask_rate] = 0  # masked
        mask[mask > self.args.mask_rate] = 1  # remained
        inp = batch_x.masked_fill(mask == 0, 0)

        outputs = self.model(inp, batch_x_mark, None, None, mask)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, :, f_dim:]

        # add support for MS
        batch_x = batch_x[:, :, f_dim:]
        mask = mask[:, :, f_dim:]

        return outputs, batch_x, mask

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Do training step.

        :param batch: input data for model
        :param batch_idx: batch index (need for PL)
        :return: loss on training data
        """
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        
        outputs, batch_x, mask = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
        loss = self.loss_function(outputs[mask == 0], batch_x[mask == 0])
                
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.train_loss.append(loss.detach().cpu().numpy())
            
        return loss
    

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Do validation step.

        :param batch: input data for model
        :param batch_idx: batch index (need for PL)
        :return: dict with loss, accuracy, target and predictions
        """
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        outputs, batch_x, mask = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

        pred = outputs.detach().cpu()
        true = batch_x.detach().cpu()
        mask = mask.detach().cpu()

        loss = self.loss_function(pred[mask == 0], true[mask == 0])

        self.log("val_loss", loss, prog_bar=True)
        self.val_step_outputs.append({
            "val_loss": loss,
            "val_target": true[mask == 0],
            "val_predictions": pred[mask == 0],
        
        })
        
        self.val_loss.append(loss)
        
        return {
            "val_loss": loss,
            "val_target": true[mask == 0],
            "val_predictions": pred[mask == 0],
        }
    
    def test_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Do validation step.

        :param batch: input data for model
        :param batch_idx: batch index (need for PL)
        :return: dict with loss, accuracy, target and predictions
        """
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        outputs, batch_x, mask = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
        pred = outputs.detach().cpu()
        true = batch_x.detach().cpu()
        mask = mask.detach().cpu()
        
        loss = self.loss_function(pred[mask == 0], true[mask == 0])

        self.log("test_loss", loss, prog_bar=True)
        self.test_step_outputs.append({
            "test_loss": loss,
            "test_target": true[mask == 0],
            "test_predictions": pred[mask == 0],
        })
        return {
            "test_loss": loss,
            "test_target": true,
            "test_predictions": pred,
        }


    def on_train_epoch_end(self):
        self.train_loss_log.append(np.mean(self.train_loss))
        self.train_loss = []
        


    def on_validation_epoch_end(self) -> None:
        """Calculate accuracy, ROC_AUC, PR_AUC after epoch end.

        :param outputs: dict with loss, accuracy, target and predictions
        """

        preds = torch.cat(tuple([x["val_predictions"].reshape(-1) for x in self.val_step_outputs]))
        trues = torch.cat(tuple([x["val_target"].reshape(-1) for x in self.val_step_outputs]))
        preds, trues = preds.numpy(), trues.numpy()
        

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        log_dict = {
            "val_mae": mae,
            "val_mse": mse,
            "val_rmse": rmse,
            "val_mape": mape,
            "val_mspe": mspe,
        }

        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True)

        
        self.val_loss_log.append(np.mean(self.val_loss))
        self.val_loss = []

        self.val_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """Calculate accuracy, ROC_AUC, PR_AUC after epoch end.

        :param outputs: dict with loss, accuracy, target and predictions
        """

        preds = torch.cat(tuple([x["test_predictions"].reshape(-1) for x in self.test_step_outputs]))
        trues = torch.cat(tuple([x["test_target"].reshape(-1) for x in self.test_step_outputs]))
        preds, trues = preds.numpy(), trues.numpy()
        self.test_step_outputs.clear()

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        log_dict = {
            "test_mae": mae,
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mape": mape,
            "test_mspe": mspe,
        }

        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Set parameters for optimizer.

        :return: optimizer
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        # scheduler1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=self.args["sch_start_factor"])
        # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.args["sch_gamma"])
        # scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[scheduler1, scheduler2], milestones=[self.args["n_warmup"]])
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        
        return {'optimizer': opt,'lr_scheduler':scheduler}
        
    

    def _get_data(self, flag):
        _, data_loader = data_provider(self.args, flag)
        return data_loader

    def train_dataloader(self) -> DataLoader:
        train_dataloader = self._get_data(flag='train')
        self.train_inverse_transform = train_dataloader.dataset.inverse_transform
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = self._get_data(flag='val')
        self.val_inverse_transform = val_dataloader.dataset.inverse_transform
        return val_dataloader
    
    def test_dataloader(self) -> DataLoader:
        test_dataloader = self._get_data(flag='test')
        return test_dataloader

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        params = torch.load(path)
        self.model.load_state_dict(params)

    def update_args(self, args):
        self.args.__dict__ = args.__dict__.copy()
        
    def rebuild_model(self):
        self.model = self._build_model()
        
        



