# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from multiprocessing import cpu_count, Pool
from ann import *
import cloudpickle as cp

if __name__ == "__main__":
    

    # load training database
    db = np.load('../notebooks/pqhanndb.npy')
    print(db[0])

    # format database
    db = Data(db, [0, 1, 2, 3, 4], [5])
    

    # net trainer
    def train_nets(net):
        net.train(db.i[:20000], db.o[:20000], epo=2000, lr=1e-4, gpu=False, ptst=0.1)
        return net

    # try to find previous neural networks
    try:
        nets = cp.load(open("pendulum_nets.p", "rb"))
        #db   = cp.load(open("pendulum_db.p", "rb"))
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

        nets = [Pendulum_Controller([8] + [shape[0]]*shape[1] + [3]) for shape in shapes]
        print("Created nets")

    # train
    nets = Pool(cpu_count()).map(train_nets, nets)

    # save nets
    cp.dump(nets, open("pendulum_nets.p", "wb"))
    cp.dump(db, open("pendulum_db.p", "wb"))