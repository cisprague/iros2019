# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import torch, numpy as np

class MLP(torch.nn.Sequential):

    def __init__(self, shape):

        torch.manual_seed(0)

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
            
            self.ops.append(torch.nn.LayerNorm(self.shape[i]))

            # linear layer
            self.ops.append(torch.nn.Linear(self.shape[i], self.shape[i + 1]))

            # if penultimate layer
            if i == self.nl - 2:

                # output between 0 and 1
                self.ops.append(torch.nn.Tanh())
                pass

            # if hidden layer
            else:

                # activation
                self.ops.append(torch.nn.Softplus())
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
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

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
            print("ANN {}; Episode {}; Testing Loss {}; Training Loss {}".format(self.shape, e, self.ltst[-1], self.ltrn[-1]))

            # backpropagate training error
            ltrn.backward()

            # update weights
            opt.step()



    def trainy(self, idat, odat, epo=500, batches=1, lr=1e-4, ptst=0.1, gpu=False):

        # put the net on the gpu if needed
        if gpu:
            self.cuda(async=True)
        else:
            self.cpu()

        # number of testing samples
        ntst = int(ptst*idat.shape[0])

        # optimiser
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        # loss function
        lf = torch.nn.MSELoss()

        # seperate training data into batches
        input_train = idat[ntst:].chunk(batches)
        output_train = odat[ntst:].chunk(batches)

        # seperate testing data into batches
        input_test = idat[:ntst].chunk(batches)
        output_test = odat[:ntst].chunk(batches)

        # for each episode
        for e in range(epo):

            # zero the gradient
            opt.zero_grad()

            # episode loss
            msetrn = 0
            msetst = 0

            # for each batch
            for itrn, otrn, itst, otst in zip(input_train, output_train, input_test, output_test):

                # put on gpu if needed
                if gpu:
                    itrn = itrn.cuda()
                    itst = itst.cuda()
                    otrn = otrn.cuda()
                    otst = otst.cuda()

                # training loss
                ltrn = lf(self(itrn), otrn)

                # testing loss
                ltst = lf(self(itst), otst)

                # accumulate gradient
                ltrn.backward()

                # track losses
                msetrn += ltrn.item()
                msetst += ltst.item()

            # update weights
            opt.step()

            # record losses
            self.ltrn.append(msetrn/batches)
            self.ltst.append(msetst/batches)

            # message
            print("ANN {}; Episode {}; Testing Loss {}; Training Loss {}".format(self.shape, e, self.ltst[-1], self.ltrn[-1]))









class Data(object):

    def __init__(self, data, cin, cout):

        # cast as numpy array
        data = np.array(data)

        # shuffle rows
        np.random.shuffle(data)

        # number of samples
        self.n = data.shape[0]

        # cast to torch
        self.i = torch.from_numpy(data[:, cin]).double().share_memory_()
        self.o = torch.from_numpy(data[:, cout]).double().share_memory_()

class Pendulum_Controller(MLP):

    def __init__(self, shape):

        # initialise MLP
        MLP.__init__(self, shape)

    def predict(self, s, alpha):
        sa = np.hstack((s, [alpha]))
        sa = torch.from_numpy(sa).double()
        sa = self(sa).detach().numpy().flatten()[0]
        return sa

class Spacecraft_Controller(MLP):

    def __init__(self, shape):

        # initialise MLP
        MLP.__init__(self, shape)

    def __call__(self, x):

        # original output
        x = MLP.__call__(self, x)

        # throttle
        u = (x[:,0] + 1)/2

        # thrust inclination [0, pi]
        theta = (x[:,1] + 1)/2*np.pi

        # thrust azimuth
        phi = (x[:,2] + 1)/2*np.pi*2

        # convert to rene descartes
        ux = torch.sin(theta)*torch.cos(phi)
        uy = torch.sin(theta)*torch.sin(phi)
        uz = torch.cos(theta)

        return torch.stack((u, ux, uy, uz), dim=1)

    def predict(self, s, alpha):
        sa = np.hstack((s, [alpha]))
        sa = np.array([sa])
        sa = torch.from_numpy(sa).double()
        sa = self(sa).detach().numpy().flatten()
        return sa

class Spacecraft_Direction_Controller(MLP):

    def __init__(self, shape):

        # initialise MLP
        MLP.__init__(self, shape)

    def __call__(self, x):

        # original output
        x = MLP.__call__(self, x)

        # thrust inclination [0, pi]
        theta = (x[:,0] + 1)/2*np.pi

        # thrust azimuth
        phi = (x[:,1] + 1)/2*np.pi*2

        # convert to rene descartes
        ux = torch.sin(theta)*torch.cos(phi)
        uy = torch.sin(theta)*torch.sin(phi)
        uz = torch.cos(theta)

        return torch.stack((ux, uy, uz), dim=1)

class Spacecraft_Thrust_Controller(MLP):

    def __init__(self, shape):

        # initialise MLP
        MLP.__init__(self, shape)

    def __call__(self, x):

        # original output
        x = MLP.__call__(self, x)

        # thrust throttle
        u = (x[:,0] + 1)/2

        return u

class Spacecraft_Controller_Joined(object):

    def __init__(self, throttle, direction):

        # sub neural networks
        self.throttle = throttle
        self.direction = direction

    def predict(self, s, alpha):

        # compute throttle
        sa = np.hstack((s, [alpha]))
        sa = np.array([sa])
        sa = torch.from_numpy(sa).double()
        u = self.throttle(sa).detach().numpy().flatten()[0]

        # compute direction
        s = np.array([s])
        s = torch.from_numpy(s).double()
        ux, uy, uz = self.direction(s).detach().numpy().flatten()

        # return complete control
        return np.array([u, ux, uy, uz])







