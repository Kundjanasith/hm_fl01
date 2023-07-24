import utils 

communication_rounds = 50
num_clients = 5

model = utils.model_init()
model.save_weights('global_models/model_ep0.h5')

for r in range(1,communication_rounds):
    print('ROUND: ',r)
    model_paths = []
    model_paths.append('global_models/model_ep%d.h5'%(r-1))
    for n in range(num_clients):
        model = utils.model_init()
        model.load_weights('global_models/model_ep%d.h5'%(r-1))
        X_train, Y_train = utils.sampling_data(1000)
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, Y_train,epochs=5,batch_size=16,verbose=1,validation_split=0.2)
        model.save_weights('local_models/n%d_model_ep%d.h5'%(n,r))
        model_paths.append('local_models/n%d_model_ep%d.h5'%(n,r))
    aggregated_model = utils.aggregated(model_paths)
    aggregated_model.save_weights('global_models/model_ep%d.h5'%(r))

