import click
import glob
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from ivbase.utils.commons import is_dtype_torch_tensor, to_tensor
from ivbase.utils.gradcheck import GradFlow
from ivbase.utils.gradinspect import GradientInspector
from ivbase.utils.metrics import accuracy, roc_auc_score
from ivbase.utils.timer import timeit
from ivbase.utils.trainer import Trainer
from pytoune.framework.callbacks import *
from pytoune.framework.iterators import EpochIterator, StepIterator, _get_step_iterator

from tensorboardX import SummaryWriter

from seq3seq.data import collate_func ,load_config, load_dataset, read_data, tensor2string
from seq3seq.const import VOCAB
from seq3seq import Seq3Seq, loss_fn

#CLIP = 2
#logging.basicConfig(level=logging.DEBUG)

def test(trainer, generator, *, steps=None):
    if steps is None and hasattr(generator, '__len__'):
        steps = len(generator)
    pred_y = []
    dec_smiles = []
    trainer.model.eval()
    with torch.no_grad():
        for _, (x, *y) in _get_step_iterator(steps, generator):
            x, *y = trainer._process_input(x, *y)
            y_pred, (_, xhat) = trainer.model(x, teacher_forcing_ratio=0)
            pred_y.append(y_pred)
            smiles = [tensor2string(xhat[i]) for i in range(xhat.size(0))]
            dec_smiles.extend(smiles)
    return np.concatenate(pred_y), dec_smiles


@timeit(log=True)
@click.command()
@click.argument("config")
@click.option('--seed', default=42, type=click.INT, help="Random seed generator")
@click.option('--inspect', is_flag=True, help="Whether to inspect gradient flow")
@click.option('--epoch',  type=click.INT, default=10, help="Number of epochs")
@click.option('--batch',  type=click.INT, default=32, help="Batch size")
def main(config, seed, inspect, epoch, batch):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    out_size = 2
    # Read the data, transform into adjacency matrices, and split into train/valid/test datasets.
    smiles, y = read_data('expts/tox21_micro.csv', num_columns=out_size)
    #X = add_eigenvectors(X, num_eig=num_eig)
    train_dt, valid_dt, test_dt = load_dataset(smiles, y, test_size=0.2, valid_size=0.2, balance=True)

    config = load_config(config)
    metrics = {'acc': accuracy}
    network = Seq3Seq(len(VOCAB)+1, config["embedding"], out_size, config) # do not forget padding

    trainer = Trainer(network, loss_fn=loss_fn, metrics=metrics, op__lr=1e-3) # Define the trainer

    if inspect:
        callbacks = [GradientInspector(top_zoom=0.2, update_at="batch")]
        #callbacks = [GradFlow(outfile=outfile, enforce_sanity=False, max_val=1e4)]
    else:
        callbacks = []

    #callbacks.append(ClipNorm(network.parameters(), CLIP))
    callbacks.append(EarlyStopping())

    log_path = os.path.join(trainer.model_dir, "metrics.csv")
    logger = CSVLogger(log_path, batch_granularity=False, separator='\t')
    callbacks += [logger]

    tboardX = {"log_dir" : os.path.join(trainer.model_dir, ".logs")}
    trainer.writer = SummaryWriter(**tboardX) # this has a purpose, which is to access the writer from the GradFlow Callback
    callbacks += [TensorBoardLogger(trainer.writer)]

    # Initialize the training parameters
    train_generator = DataLoader(train_dt, batch_size=batch, collate_fn=collate_func, shuffle=True)
    valid_generator = DataLoader(valid_dt, batch_size=batch, collate_fn=collate_func, shuffle=True)
    test_generator = DataLoader(test_dt, batch_size=batch, collate_fn=collate_func, shuffle=True)

    trainer.fit_generator(train_generator, valid_generator, epochs=epoch, callbacks=callbacks)

    y_pred, pred_smiles = test(trainer, test_generator)

    acc = accuracy(y_pred, test_dt.y, test_dt.w)
    roc_auc = roc_auc_score(y_pred, test_dt.y)   # Compute the ROC-AUC score on the testing set
    print('Score test \tacc: ', acc, '\troc_auc: ', roc_auc) # Print the results
    for sm, pred_sm in zip(test_dt.smiles, pred_smiles):
        print(sm, "==>", pred_sm)

if __name__== "__main__":
    main()