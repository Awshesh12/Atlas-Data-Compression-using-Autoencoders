# Running the network, training and validation

n_features = len(train.loc[0])
model = AE_3D()

epochs_list = [50,100,100]
lrs = [2e-2,1e-3,4e-4]
for ii, epochs in enumerate(epochs_list):
    print('Setting learning rate to %.1e' % lrs[ii])
    opt = optim.Adam(model.parameters(), lr=lrs[ii])
    b=fit(epochs, model, loss_func, opt, train_dl, valid_dl,device)

