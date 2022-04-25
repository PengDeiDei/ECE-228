from __future__ import print_function, division
# from future import standard_library
# standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle
from .data_utils import Data
from . import optim
import cv2
import numpy as np

class Solver(object):
    def __init__(self, model, **kwargs):
        self.model = model
        self.update_rule =  kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        self.dataset_directory = kwargs.pop('dataset_directory', None)

        self.data_processor = Data(self.batch_size, self.dataset_directory)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if isinstance(self.update_rule, str):
            if not hasattr(optim, self.update_rule):
                raise ValueError('Invalid update_rule "%s"' % self.update_rule)
            self.update_rule = getattr(optim, self.update_rule)
        else:
            assert callable(self.update_rule), 'The update rule is not callable'

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
    
    def _step(self, inputs, labels):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Compute loss and gradient
        loss, grads = self.model.loss(inputs, labels)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config
    
    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
            'model': self.model,
            'update_rule': self.update_rule,
            'lr_decay': self.lr_decay,
            'optim_config': self.optim_config,
            'batch_size': self.batch_size,
            'num_train_samples': self.num_train_samples,
            'num_val_samples': self.num_val_samples,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def check_accuracy(self, dataloader):
        y_pred = []
        y = []
        for batch, (inputs, labels) in enumerate(dataloader, 0):
            scores = self.model.loss(inputs)
            y.append(labels)
            y_pred.append(np.argmax(scores, axis=1))
        # print(labels, np.max(scores, axis=1))
        y_pred = np.hstack(y_pred)
        y = np.hstack(y)
        acc = np.mean(y_pred == y)
        
        return acc*100
    
    def train(self):
        """
        Run optimization to train the model.
        """

        for epoch in range(self.num_epochs):
            trainloader = self.data_processor.load_train_data()

            for batch, (inputs, labels) in enumerate(trainloader, 0):
                self._step(inputs, labels)
                # Maybe print training loss
                if self.verbose and batch % self.print_every == 0:
                    print('(Epoch %d Batch %d) loss: %f' % (epoch, batch+1, self.loss_history[-1]))

            train_acc = self.check_accuracy(self.data_processor.load_train_data())
            val_acc = self.check_accuracy(self.data_processor.load_val_data())
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            self._save_checkpoint()

            if self.verbose:
                print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (epoch, self.num_epochs, train_acc, val_acc))

            # Keep track of the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            self.best_params = {}
            for k, v in self.model.params.items():
                self.best_params[k] = v.copy()

            # At the end of training swap the best params into the model
            self.model.params = self.best_params
            for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
