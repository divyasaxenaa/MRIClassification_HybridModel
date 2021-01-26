import matplotlib.pyplot as plt


def graphplt(x_value1,x_value2,x_value3,x_value4, fig_main, labelX, labelY):
      plt.figure(figsize=[15, 15],num=fig_main)
      plt.plot(x_value1,color='olive',label = "GCN")
      plt.plot(x_value2, color='red', label="GCN+CNN")
      plt.plot(x_value3, color='blue', label="CNN")
      plt.plot(x_value4, color='indigo', label="GCN+RNN")
      plt.xlabel(labelX, fontsize=14)
      plt.ylabel(labelY, fontsize=14)
      plt.legend(loc='best')
      plt.savefig(fig_main+".png")
      plt.show()
