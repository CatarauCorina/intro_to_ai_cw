import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import data_preproc as dp


class PlasticCNN(nn.Module):

    def __init__(self, params):

        super(PlasticCNN, self).__init__()
        self.rule = params['rule']
        self.learning_rate = params['lr']
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.cv1 = torch.nn.Conv2d(3, 64, 3, stride=2)
        self.cv2 = torch.nn.Conv2d(64, 64, 3, stride=2)
        self.cv3 = torch.nn.Conv2d(64, 64, 3, stride=2)
        self.cv4 = torch.nn.Conv2d(64, 64, 3, stride=2)

        self.w = torch.nn.Parameter((
                self.learning_rate * torch.randn(params['nbf'], params['nr_classes'])),
            requires_grad=True)
        if params['alpha']:
            self.alpha = torch.nn.Parameter((self.learning_rate * torch.rand(params['nbf'], params['nr_classes'])),
                                            requires_grad=True)
        else:
            self.alpha = torch.nn.Parameter((self.learning_rate* torch.ones(1)),
                                            requires_grad=True)

        self.eta = torch.nn.Parameter((self.learning_rate * torch.ones(1)),
                                      requires_grad=True)
        self.params = params
        return

    def forward(self, input, in_label, hebb_trace):
        activation_function = getattr(F, self.params['activation'])
        activ = activation_function(self.cv1(input))
        activ = activation_function(self.cv2(activ))
        activ = activation_function(self.cv3(activ))
        activ = activation_function(self.cv4(activ))
        activ_in = activ.view(-1, self.params['nbf'])

        z_out = activ_in.mm(self.w + torch.mul(self.alpha, hebb_trace)) + Variable(in_label, requires_grad=False)
        activ_out = F.softmax(z_out)
        hebb_trace = self.update_hebb_trace(
            self.learning_rate,
            activ_in.unsqueeze(2),
            activ_out.unsqueeze(1),
            hebb_trace
        )
        return activ_out, hebb_trace

    def init_hebb_trace(self):
        ttype = torch.FloatTensor
        hebb_trace = Variable(torch.zeros(self.params['nbf'], self.params['nr_classes']).type(ttype))
        return hebb_trace

    def update_hebb_trace(self, lr, yi, yj, prev_hebb):
        return (1 - lr) * prev_hebb + lr * torch.bmm(yi, yj)[0]


class Plasticity:

    def __init__(self):
        self.activation = 'tanh'
        self.update_rule = 'hebb'
        self.learnable_plasticity = True
        # self.nr_classes = y_shape
        self.pattern_shot = 1  # Number of times each pattern is to be presented
        self.presentation_delay = 0  # Duration of zero-input interval between presentations
        # self.image_size = x_shape
        self.learning_rate = 3e-5
        self.nr_iterations = 10000
        self.test_every = 500
        self.nb_episodes = 6
        plastic_celeb_data = dp.PlasticDataCreator()
        self.plastic_celeb_data = plastic_celeb_data
        self.nr_classes = len(pd.Series(self.plastic_celeb_data.celeb_loader.train_targets).unique())

        return

    def test_time(self, num_iter, y, target):
        print(num_iter, "====")
        td = target.cpu().numpy()
        yd = y.data.cpu().numpy()[0]
        print("y: ", yd[:10])
        print("target: ", td[:10])
        error = np.abs(td - yd)
        print(f'Mean {np.mean(error)} / median {np.median(error)} / max abs diff {np.max(error)}')
        print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])
        return

    def train(self, params):
        net = PlasticCNN(params)
        optimizer = torch.optim.Adam(net.parameters(), lr=1.0 * params['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=params['gamma'],
            step_size=params['steplr']
        )
        all_losses_objective = []
        lossbetweensaves = 0.0

        for iteration in range(self.nr_iterations):
            hebb_trace = net.init_hebb_trace()
            optimizer.zero_grad()
            is_test_step = ((iteration + 1) % self.test_every == 0)
            inputs, labels, test_labels = self.plastic_celeb_data.create_input_plastic_network(type_ds='train')

            for num_step in range(self.nb_episodes):
                y, hebb_trace = net(
                    Variable(inputs[num_step], requires_grad=False),
                    Variable(labels[num_step], requires_grad=False), hebb_trace)

            criterion = torch.nn.BCELoss()
            loss = criterion(y[0], Variable(test_labels, requires_grad=False))
            if is_test_step == False:
                loss.backward()
                maxg = 0.0
                scheduler.step()
                optimizer.step()
            loss_value = float(loss.data)
            lossbetweensaves += loss_value
            all_losses_objective.append(loss_value)
            if is_test_step:
                self.test_time(iteration, y, test_labels)


if __name__ == "__main__":
    plasticity = Plasticity()
    plasticity.train({
        'lr': 0.01,
        'gamma': .666,
        'nbf': 64,
        'rule': 'hebb',
        'nr_classes': plasticity.nr_classes,
        'alpha': True,
        'steplr': 1e6,
        'activation': 'tanh'
    })

