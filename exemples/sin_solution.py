from DeepLib import *
import matplotlib.pyplot as plt

neural_net = DNN()

neural_net.load("./models/sin_definitive")

X = np.linspace(0,2*np.pi,10000)
Y = np.sin(X)

def validar(rede,XTest,YTest):
            Acuracy = 0
            Total = len(XTest)
            for x,y in zip(XTest,YTest):
                pred = rede(x)
                if abs(float((pred))-y) < 0.005:
                    Acuracy += 1
            return Acuracy/Total

#testing the model
Ypred = []
for x in X:
    Ypred.append(neural_net(x)[0][0])

acuracy = validar(neural_net,X,Y)

plt.style.use('dark_background')        
fig, comp = plt.subplots()
comp.plot(X,Y,'b',label='Real')
comp.plot(X,Ypred, 'y', label='Aproximado')
comp.text(0.05, 0.05, f'Acurácia: {(acuracy*100):.3f}%', transform=comp.transAxes, fontsize=16, verticalalignment='bottom')
comp.legend()
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.show()  



