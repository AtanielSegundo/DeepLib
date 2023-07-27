from sklearn.model_selection import train_test_split as ts
from DeepLib import *
import matplotlib.pyplot as plt

neural_net = DNN()

neural_net.set_parameter(alpha = 0.005 , batch_size = 32 , 
                         loss_function = neural_net.Huber_Loss,
                         delta = 1.0)

init = "xavier"
neural_net.add_layer(
    layers.Relu(1,200,init),
    layers.Leaky_Relu(200,100,init),
    layers.Leaky_Relu(100,50,init),
    layers.Tanh(50,1,init)
)

#Initializing data for training
X = np.linspace(0,2*np.pi,20_000)
Y = np.sin(X)

#spliting data for training
Xtrain,Xtest,Ytrain,Ytest = ts(X,Y,test_size=0.3,random_state=66)

train_zip = list(zip(Xtrain,Ytrain))
#spliting data in batchs
for epoch in (epochs:=range(50)):
    
    print("Epoch = ",epoch,end = "\r")
    
    np.random.shuffle(train_zip)
    for i in range(0,len(train_zip),neural_net.batch_size):
        batch = train_zip[i:i+neural_net.batch_size]
        Xbatch, Ybatch = zip(*batch)
        Xbatch = np.array(Xbatch)
        Ybatch = np.array(Ybatch)

        neural_net.fit(Xbatch,Ybatch,epochs = 1)

#testing the model
Ypred = []
for x in X:
    Ypred.append(neural_net(x)[0][0])

neural_net.save(path = "./models",name = "sin_go_horse")

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(X, Ypred, color='red', label='Predicted')
plt.plot(X, Y, color='blue', label='True')
plt.title('True Values vs Predicted Values')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()