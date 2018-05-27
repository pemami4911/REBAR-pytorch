import matplotlib.pyplot as plt
import h5py
import os
import seaborn as sns
sns.set()

if __name__ == '__main__':
    save_dir = '/tmp/rebar/1337'
    scores_file = os.path.join(save_dir, 'scores.hdf5')
    scores = h5py.File(scores_file, 'r')

    ELBO = []
    logGradVar = []
    for sc in scores['scores']:
        ELBO.append(sc[0])
        logGradVar.append(sc[1])

    # plot ELBO
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(ELBO)
    ax2.plot(logGradVar)

    plt.show()
