# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from torch.multiprocessing import cpu_count, Pool
from ann import *
import cloudpickle as cp
import numpy as np

if __name__ == "__main__":
    
    # load training database
    db = np.load('data/sdata.npy')
    di = torch.from_numpy(db[:, [0, 1, 2, 3, 4, 5, 6, 7]]).double()
    do = torch.from_numpy(db[:, [8, 9, 10, 11]]).double()
    

    # net trainer
    def train_nets(net):
        net.train(di, do, epo=10000, lr=1e-3, gpu=True, ptst=0.1)
        return net

    # try to find previous neural networks
    try:
        nets = cp.load(open("spacecraft_nets.p", "rb"))
        print("Found nets")

    # otherwise create them
    except:
        # neural network architectures
        shapes = [
            (50, 2),
            (50, 4),
            (100, 2),
            (100, 4)
        ]

        nets = [Spacecraft_Controller([8] + [shape[0]]*shape[1] + [3]) for shape in shapes]
        print("Created nets")

    # train
    nets = Pool(cpu_count()).map(train_nets, nets)


    # save nets
    cp.dump(nets, open("spacecraft_nets.p", "wb"))
