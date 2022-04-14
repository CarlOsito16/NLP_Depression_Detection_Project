import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, loss, batch_acc, epoch, n_batch, num_batches):

        # var_class = torch.autograd.variable.Variable
        if isinstance(loss, torch.autograd.Variable):
            loss = loss.item()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/loss'.format(self.comment), loss, step)
        self.writer.add_scalar(
            '{}/accuracy'.format(self.comment), batch_acc, step)

    # def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
    #     '''
    #     input images are expected in format (NCHW)
    #     '''
    #     if type(images) == np.ndarray:
    #         images = torch.from_numpy(images)
        
    #     if format=='NHWC':
    #         images = images.transpose(1,3)
        

    #     step = Logger._step(epoch, n_batch, num_batches)
    #     img_name = '{}/images{}'.format(self.comment, '')

    #     # Make horizontal grid from image tensor
    #     horizontal_grid = vutils.make_grid(
    #         images, normalize=normalize, scale_each=True)
    #     # Make vertical grid from image tensor
    #     nrows = int(np.sqrt(num_images))
    #     grid = vutils.make_grid(
    #         images, nrow=nrows, normalize=True, scale_each=True)

    #     # Add horizontal images to tensorboard
    #     self.writer.add_image(img_name, horizontal_grid, step)

    #     # Save plots
    #     self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    # def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
    #     out_dir = './data/images/{}'.format(self.data_subdir)
    #     Logger._make_dir(out_dir)

    #     # Plot and save horizontal
    #     fig = plt.figure(figsize=(16, 16))
    #     plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
    #     plt.axis('off')
    #     if plot_horizontal:
    #         display.display(plt.gcf())
    #     self._save_images(fig, epoch, n_batch, 'hori')
    #     plt.close()

    #     # Save squared
    #     fig = plt.figure()
    #     plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
    #     plt.axis('off')
    #     self._save_images(fig, epoch, n_batch)
    #     plt.close()

    # def _save_images(self, fig, epoch, n_batch, comment=''):
    #     out_dir = './data/images/{}'.format(self.data_subdir)
    #     Logger._make_dir(out_dir)
    #     fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
    #                                                      comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, loss, batch_acc):
        
        # var_class = torch.autograd.variable.Variable
        if isinstance(loss, torch.autograd.Variable):
            loss = loss.item()
        
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch,num_epochs, n_batch, num_batches)
             )
        print('Loss: {:.4f}'.format(loss))
        print('Accuracy: {:.4f}'.format(batch_acc))


    def save_models(self, model, epoch, num_batches):
        out_dir = './models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(model.state_dict(),
                   '{}/model_epoch_{}_num{}'.format(out_dir, epoch, num_batches))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise