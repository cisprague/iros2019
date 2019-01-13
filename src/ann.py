# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import torch, numpy as np

class MLP(torch.nn.Sequential):

    def __init__(self, shape):

        # architecture
        self.shape = shape

        # number of inputs
        self.ni = self.shape[0]

        # number of outputs
        self.no = self.shape[-1]

        # number of layers
        self.nl = len(self.shape)

        # loss data
        self.ltrn, self.ltst = list(), list()

        # operations
        self.ops = list()

        # apply operations
        for i in range(self.nl - 1):

            # linear layer
            self.ops.append(torch.nn.Linear(self.shape[i], self.shape[i + 1]))

            # batch normalisation
            #self.ops.append(torch.nn.BatchNorm1d(self.shape[i + 1]))

            # if penultimate layer
            if i == self.nl - 2:

                # output between 0 and 1
                self.ops.append(torch.nn.Tanh())
                pass

            # if hidden layer
            else:

                # activation
                self.ops.append(torch.nn.LeakyReLU())
                pass

        # initialise neural network
        torch.nn.Sequential.__init__(self, *self.ops)
        self.double()

    def train(self, idat, odat, epo=100, lr=1e-4, ptst=0.1, gpu=False):

        if gpu:
            self.cuda()
            idat = idat.cuda()
            odat = odat.cuda()
        else:
            self.cpu()
            idat = idat.cpu()
            odat = odat.cpu()

        # number of testing data points
        n = int(idat.shape[0]*ptst)

        # testing data
        itst = idat[:n, :]
        otst = odat[:n, :]

        # training data
        itrn = idat[n:, :]
        otrn = odat[n:, :]

        # optimiser
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        # loss function
        lf = torch.nn.MSELoss()

        # iterate through episodes
        for e in range(epo):

            # zero gradients
            opt.zero_grad()

            # testing loss
            ltst = lf(self(itst), otst)

            # training loss
            ltrn = lf(self(itrn), otrn)

            # record losses
            self.ltst.append(ltst.item())
            self.ltrn.append(ltrn.item())

            # print progress
            print("Episode {}; Testing Loss {}; Training Loss {}".format(e, self.ltst[-1], self.ltrn[-1]))

            # backpropagate training error
            ltrn.backward()

            # update weights
            opt.step()

class Data(object):

    def __init__(self, data, cin, cout):

        # cast as numpy array
        data = np.array(data)

        # shuffle rows
        np.random.shuffle(data)

        # number of samples
        self.n = data.shape[0]

        # cast to torch
        self.i = torch.from_numpy(data[:, cin]).double()
        self.o = torch.from_numpy(data[:, cout]).double()

class Pendulum_Controller(MLP):

    def __init__(self, shape):

        # initialise MLP
        MLP.__init__(self, shape)

    def predict(self, s, alpha):
        sa = np.hstack((s, [alpha]))
        sa = torch.from_numpy(sa).double()
        sa = self(sa).detach().numpy().flatten()[0]
        return sa


