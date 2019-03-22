import matplotlib.pyplot as plt, cloudpickle as cp, numpy as np

if __name__ == "__main__":

    # get the nets
    pnets = cp.load(open("pendulum_nets.p", "rb"))

    # create a grayscale
    gs = np.linspace(0, 0.4, len(pnets))

    # create figure
    fig, ax = plt.subplots(2, sharex=True)

    # plot pendulum nets
    for i, net in enumerate(pnets):
        ax[0].plot(net.ltrn, "-", color=str(gs[i]))
        ax[0].plot(net.ltst, "--", color=str(gs[i]))
    

    # set scales
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")


    plt.show()