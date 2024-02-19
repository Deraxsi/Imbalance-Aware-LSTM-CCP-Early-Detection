import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

df = pd.read_csv(r"C:\Users\dera0000\Downloads\T=80, prm=0.53, id=f7514131c2.csv")

losses = df.val_loss
gmeans = df.val_geometric_mean
epochs = df.epoch

# fig, ax = plt.subplots()
# ax.scatter(gmeans, losses, alpha=0.3)
# ax.set_xlabel(r"Validation G-mean $(\times 10^{-3})$")
# ax.set_ylabel(r"Validation Loss $(\times 10^{-3})$")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# ax.scatter(gmeans, losses, epochs, alpha=0.3)
# ax.set_xlabel(r"Validation G-mean $(\times 10^{-3})$")
# ax.set_ylabel(r"Validation Loss $(\times 10^{-3})$")
# ax.set_zlabel(r"No. Epochs")

# ax.scatter(gmeans, losses, epochs, alpha=0.3)
# ax.set_xlabel(r"No. Epochs")
# ax.set_ylabel(r"Validation G-mean $(\times 10^{-3})$")
# ax.set_zlabel(r"Validation Loss $(\times 10^{-3})$")

# ax.scatter(epochs, gmeans, losses, alpha=0.3)
# ax.set_xlabel(r"No. Epochs")
# ax.set_ylabel(r"Validation G-mean $(\times 10^{-3})$")
# ax.set_zlabel(r"Validation Loss $(\times 10^{-3})$")

ax.scatter(losses, gmeans, epochs, alpha=0.3)
ax.set_zlabel(r"No. Epochs")
ax.set_ylabel(r"Validation G-mean $(\times 10^{-3})$")
ax.set_xlabel(r"Validation Loss $(\times 10^{-3})$")


best_loss = 10**10
best_gmean = 0
best_epoch = 0

kappa = 2.8
flag = False
solution_id = 1
eps = 0.05

for epoch in epochs:
    loss = losses[epoch]
    gmean = gmeans[epoch]

    loss_percentage_change = (loss - best_loss) / (best_loss + 1e-20)
    gmean_percentage_change = (gmean - best_gmean) / (best_gmean + 1e-20)
    weight_updating_criterion = loss_percentage_change - kappa * gmean_percentage_change

    if weight_updating_criterion < 0:
        if epoch:
            # ax.annotate(str(epoch), xytext=(best_gmean, best_loss), xy=(gmean, loss),
            #             arrowprops=dict(arrowstyle="->", alpha=0.7, facecolor='black'))
            # ax.annotate(str(epoch), xyztext=(best_epoch, best_gmean, best_loss), xyz=(epoch, gmean, loss),
            #             arrowprops=dict(arrowstyle="->", alpha=0.7, facecolor='black'))
            # ax.text(epoch, gmean, loss, str(epoch), color='black', alpha=0.6)
            if solution_id in [2, 9]:
                # ax.text(epoch, gmean, loss + eps, str(solution_id), color='black', alpha=0.6)
                ax.text(loss, gmean + eps, epoch, str(solution_id), color='black', alpha=0.6)
            elif solution_id in [12, 14]:
                # ax.text(epoch, gmean, loss - 2 * eps, str(solution_id), color='black', alpha=0.6)
                ax.text(loss - eps, gmean + eps, epoch, str(solution_id), color='black', alpha=0.6)
            elif solution_id in [17]:
                # ax.text(epoch, gmean + 2 * eps, loss, str(solution_id), color='black', alpha=0.6)
                ax.text(loss, gmean + 2 * eps, epoch, str(solution_id), color='black', alpha=0.6)
            else:
                # ax.text(epoch, gmean + eps, loss + eps, str(solution_id), color='black', alpha=0.6)
                ax.text(loss, gmean + eps, epoch, str(solution_id), color='black', alpha=0.6)
            # pass

            best_epoch = epoch
            best_loss = loss
            best_gmean = gmean
            solution_id += 1

        # if epoch:
        #     # ax.annotate(str(epoch), xy=(best_loss, best_gmean), xytext=(best_loss, best_gmean),
        #     #             arrowprops=dict(facecolor='green', shrink=0.05), xycoords='data',)
        #     ax.scatter(gmean, loss, color='green')
        #     ax.text(gmean, loss, str(solution_id))
        #     plt.show()

        # ax.scatter(gmean, loss, color='green')
        # ax.text(gmean + eps, loss + eps, str(solution_id), alpha=0.4)

            # ax.scatter(epoch, gmean, loss, color='green')
            ax.scatter(loss, gmean, epoch, color='green')


        # plt.show()

max_gm = df[df.val_geometric_mean == df.val_geometric_mean.max()]
ax.scatter(max_gm.val_loss, max_gm.val_geometric_mean, max_gm.epoch, color='yellow')

min_vloss = df[df.val_loss == df.val_loss.min()]
ax.scatter(min_vloss.val_loss, min_vloss.val_geometric_mean, min_vloss.epoch, color='yellow')


plt.savefig("early_stopping_vis.pdf")
plt.show()
print('hi')



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

df = pd.read_csv(r"C:\Users\dera0000\Downloads\T=80, prm=0.53, id=f7514131c2.csv")

losses = df.val_loss
gmeans = df.val_geometric_mean
epochs = df.epoch

fig, ax = plt.subplots()
ax.scatter(gmeans, losses, alpha=0.3)
ax.set_xlabel("Validation G-mean")
ax.set_ylabel("Validation Loss")


best_loss = 10**10
best_gmean = 0
best_epoch = 0

kappa = 2.8
flag = False
solution_id = 0
eps = 20 * 1e-4

for epoch in epochs:
    loss = losses[epoch]
    gmean = gmeans[epoch]

    loss_percentage_change = (loss - best_loss) / (best_loss + 1e-20)
    gmean_percentage_change = (gmean - best_gmean) / (best_gmean + 1e-20)
    weight_updating_criterion = loss_percentage_change - kappa * gmean_percentage_change

    if weight_updating_criterion < 0:
        if epoch:
            # ax.annotate(str(epoch), xytext=(best_gmean, best_loss), xy=(gmean, loss),
            #             arrowprops=dict(arrowstyle="->", alpha=0.7, facecolor='black'))
            # ax.annotate(str(epoch), xyztext=(best_epoch, best_gmean, best_loss), xyz=(epoch, gmean, loss),
            #             arrowprops=dict(arrowstyle="->", alpha=0.7, facecolor='black'))
            # ax.text(epoch, gmean, loss, str(epoch), color='black', alpha=0.6)
            # if solution_id in [2, 9]:
            #     # ax.text(epoch, gmean, loss + eps, str(solution_id), color='black', alpha=0.6)
            #     ax.text(loss, gmean + eps, epoch, str(solution_id), color='black', alpha=0.6)
            # elif solution_id in [12, 14]:
            #     # ax.text(epoch, gmean, loss - 2 * eps, str(solution_id), color='black', alpha=0.6)
            #     ax.text(loss - eps, gmean + eps, epoch, str(solution_id), color='black', alpha=0.6)
            # elif solution_id in [17]:
            #     # ax.text(epoch, gmean + 2 * eps, loss, str(solution_id), color='black', alpha=0.6)
            #     ax.text(loss, gmean + 2 * eps, epoch, str(solution_id), color='black', alpha=0.6)
            # else:
            #     # ax.text(epoch, gmean + eps, loss + eps, str(solution_id), color='black', alpha=0.6)
            #     ax.text(loss, gmean + eps, epoch, str(solution_id), color='black', alpha=0.6)
            # pass
            if solution_id in []:
                #
                3

            ax.text(gmean + eps, loss, str(solution_id + 1), color='black', alpha=0.6)

            if solution_id:
                ax.annotate("", xytext=(best_gmean, best_loss), xy=(gmean, loss),
                            arrowprops=dict(arrowstyle="->", alpha=0.7, facecolor='black'))


            best_epoch = epoch
            best_loss = loss
            best_gmean = gmean
            solution_id += 1

        # if epoch:

        #     ax.scatter(gmean, loss, color='green')
        #     ax.text(gmean, loss, str(solution_id))
        #     plt.show()

        # ax.scatter(gmean, loss, color='green')
        # ax.text(gmean + eps, loss + eps, str(solution_id), alpha=0.4)

            # ax.scatter(epoch, gmean, loss, color='green')
            ax.scatter(gmean, loss, color='green')


        # plt.show()

max_gm = df[df.val_geometric_mean == df.val_geometric_mean.max()]
ax.scatter(max_gm.val_geometric_mean, max_gm.val_loss, color='yellow')

min_vloss = df[df.val_loss == df.val_loss.min()]
ax.scatter(min_vloss.val_geometric_mean, min_vloss.val_loss, color='yellow')


plt.savefig("early_stopping_vis_2d.pdf")
plt.show()
print('hi')


