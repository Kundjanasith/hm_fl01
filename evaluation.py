import utils

communication_rounds = 50
num_clients = 5

(X_train, Y_train), (X_test, Y_test) = utils.load_dataset()

# GLOBAL MODEL
file_o = open('global_acc.csv','w')
for r in range(communication_rounds):
    model = utils.model_init()
    model.load_weights('global_models/model_ep%d.h5'%(r))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    res = model.evaluate(X_test, Y_test)
    file_o.write(str(res[1])+'\n')

# LOCAL MODEL
file_o = open('local_acc.csv','w')
for r in range(1,communication_rounds):
    ra = ''
    for n in range(num_clients):
        model = utils.model_init()
        model.load_weights('local_models/n%d_model_ep%d.h5'%(n,r))
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        res = model.evaluate(X_test, Y_test)
        ra = ra + str(res[1])+','
    file_o.write(ra[:-1]+'\n')


