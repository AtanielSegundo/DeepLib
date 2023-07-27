#%Necessary imports for library to work
import copy
import numpy as np
import random as rand
from collections import deque

class layers:
    class Layer:
    
        def load_initializer(self,input_size,output_size):
            self.W = None
            self.B = None
            return
        
        def he_initializer(self,input_size,output_size):
            "Inicialzição de pesos e vieses pelo metodo de kaiming"
            he_normalizer = np.sqrt(2 / input_size)
            self.W = np.random.uniform(0,he_normalizer,size=(input_size,output_size)) 
            self.B = np.zeros((1,output_size))
            
        def xavier_initializer(self,input_size,output_size):     
            "Inicialzição de pesos e vieses pelo metodo de xavier"
            limit = np.sqrt(6/(input_size + output_size))
            self.W = np.random.uniform(-limit,limit,size=(input_size,output_size)) 
            self.B = np.zeros((1,output_size))
                
        def std_initializer(self,input_size,output_size):
            "Inicialzição de pesos e vieses randomicos com media 0"
            self.W = np.random.rand(input_size, output_size) - 0.5
            self.B = np.random.rand(1, output_size) - 0.5 
            
        def __init__(self,input_size,output_size,initializer,act_func,der):
            "Inicializa os pesos da rede e vieses"
            getattr(self,initializer+"_initializer")(input_size,output_size)
            self.set_activation(act_func,der)
        
        def set_activation(self,activation_function, derivative):
            "Metodo de atribuição da função de ativação ao layer(camada)"
            self.Act_func = activation_function
            self.Der = derivative
        
        def __call__(self,Input):
            "Metodo de propagação foward da camada"
            self.input  = Input 
            self.output = np.dot(self.input,self.W) + self.B
            return self.Act_func(self.output)
        
        def backpropagation(self,output_gradient,alpha):
            "Metodo de retropropagação não parametrica"
            # O operador "*" é usado para multiplicação elemento a elemento
            out_grad_act = output_gradient * self.Der(self.output)
            dW = alpha * np.dot(self.input.T,out_grad_act)
            dB = alpha * out_grad_act
            dX = np.dot(out_grad_act,self.W.T)
            return dX,dW,dB
    
#%% Classes com as funções de ativação:
    
    class Relu(Layer):

        def act_func(self, Input):
            return np.maximum(0, Input)

        def der(self, Input):
            return np.where(Input > 0, 1, 0)

        def __init__(self, input_size, output_size,init):        
            super().__init__(input_size,output_size,init,self.act_func, self.der)

    class Sigmoid(Layer):

        def act_func(self, Input):
            return 1 / (1 + np.exp(-Input))

        def der(self, Input):
            sig = self.act_func(Input)
            return sig * (1 - sig)

        def __init__(self, input_size, output_size,init):        
            super().__init__(input_size,output_size,init,self.act_func, self.der)

    class Linear(Layer):

        def act_func(self, Input):
            return Input

        def der(self, Input):
            return np.ones_like(Input)

        def __init__(self, input_size, output_size,init):        
            super().__init__(input_size,output_size,init,self.act_func, self.der)

    class Softsign(Layer):

        def act_func(self, Input):
            return Input / (1 + np.abs(Input))

        def der(self, Input):
            return 1 / (1 + np.abs(Input)) ** 2

        def __init__(self, input_size, output_size,init):        
            super().__init__(input_size,output_size,init,self.act_func,self.der)

    class Leaky_Relu(Layer):

        def act_func(self, Input):
            return np.where(Input > 0, Input, 0.01 * Input)

        def der(self, Input):
            return np.where(Input > 0, 1, 0.01)

        def __init__(self, input_size, output_size,init):        
            super().__init__(input_size,output_size,init,self.act_func, self.der)

    class Tanh(Layer):

        def act_func(self, Input):
            return (2/(1 + np.exp(Input * -2))) - 1

        def der(self, Input):
            return 1 - np.power((2 / (1 + np.exp(Input * -2))) - 1, 2)

        def __init__(self, input_size, output_size,init):        
            super().__init__(input_size,output_size,init,self.act_func, self.der)

class DNN:
    
    def save(self,path="./models",name = "no_named"):
        try:
            import os
        except:
            print("CANNOT IMPORT OS MODULE")
            return
        
        if not os.path.isdir(path):
            os.mkdir(path)

        model_dir = path+"/"+name 
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        idx = 1
        for layer in self.network:
            input_size,output_size = layer.W.shape
            layer_name = str(type(layer)).split("'")[1].split(".")[2]
            
            with open(f"{model_dir}/{idx}.{layer_name}_{input_size}_{output_size}.npy","wb") as f:
                np.save(f,layer.W)
                np.save(f,layer.B)
            idx+=1
        
        #Saving variables used in the model
        btin_classes = [int,str,float,complex,float,bool]
        cfg_array = [attr for attr in dir(self) if not callable(getattr(self, attr)) 
                     and not attr.startswith("__")]
        model_cfg = {}
        model_cfg["loss_function"] = "self."+self.loss_function.__name__
        for attr in cfg_array:
            val = getattr(self,attr)
            attr_type = type(val)
            if attr_type in btin_classes:
                model_cfg[attr] = val 
            elif attr_type == dict:
                aux_dict = {}
                for key,value in val.items():
                    if type(value) != dict and type(value) != bool:
                        aux_dict[key] = value
                model_cfg[attr] = aux_dict

        with open(model_dir+"/"+name+"_configs.txt","w") as f:
            f.write("{\n")
            for key,value in model_cfg.items():
                f.write(f"'{key}':{value},\n")
            f.write("'loaded':True\n")
            f.write("}")

    def load(self, path,load_cfg=True):
        try:
            import os
        except:
            print("CANNOT IMPORT OS MODULE")

        saved_layers = [ i for i in os.listdir(path) if i.endswith(".npy")]
        saved_layers.sort()
        
        for layer in saved_layers:
            layer_ext = layer[0:len(layer) - 4]
            *layer_name,input_size,output_size = layer_ext.split("_")
            layer_name[0] = layer_name[0].split(".")[1]
            temp = layer_name[0]
            layer_name = layer_name[1:len(layer_name)]
            for string in layer_name:
                temp+="_"+string
            layer_name = temp
            input_size =  int(input_size)
            output_size = int(output_size)

            layer_now = getattr(layers,layer_name)(input_size,output_size,"load")

            with open(path+"/"+layer,"rb") as l:
                layer_now.W = np.load(l)
                layer_now.B = np.load(l)

            self.network.append(layer_now)

        if load_cfg:
            model_name = path.split("/")[-1]
            cfg_file = open(path+"/"+model_name+"_configs.txt","r")
            cfg_str  = cfg_file.read()
            cfg_dict = eval(cfg_str)
            self.set_parameter(**cfg_dict)
            self.adam["m_tensor"] = {}
            self.adam["v_tensor"] = {}
            self.set_adam_tensors()

    def __data_treatment__(self, Arg):
        if isinstance(Arg, np.ndarray):
            return Arg
        else:
            return np.array(Arg)
    
    def MSE_Loss(self, predicted, target):
        error = (predicted - target)
        self.cost_epoch.append((0.5*(error)**2).mean())
        
        return error
    
    def MAE_Loss(self, predicted, target):
        error = predicted - target
        MAE   = np.abs(error) 
        self.cost_epoch.append(MAE.mean())
        
        return np.sign(error)
    
    def Huber_Loss(self,predicted,target):
        error = predicted - target
        abs_error = np.abs(error.mean())
        
        if abs_error <= self.delta:
            self.cost_epoch.append((0.5*(error)**2).mean())
            return error
        else:
            self.cost_epoch.append(self.delta * (abs_error - 0.5 * self.delta))
            return self.delta * np.sign(error)
         
    def set_parameter(self, **parameters):
        for var_name, value in parameters.items():
            setattr(self, var_name, value)
    
    def params_init(self):
        self.alpha = float()
        self.batch_size = int()
        self.loss_function = self.MSE_Loss
        self.network = list()
        self.cost_epoch = list()
        self.delta = 1.0
        self.adam = {
            "beta1" : 0.9,
            "beta2" : 0.999,
            "epsilon": 1e-8,
            "m_tensor": {},
            "v_tensor": {},
            "t": 0,
            "inited": False
        }

    def __init__(self,**parameters):
        self.params_init()
        self.set_parameter(**parameters)

    def set_adam_tensors(self) -> None: 
        self.adam["m_tensor"]["w"] = [np.zeros_like(layer.W) for layer in self.network]
        self.adam["m_tensor"]["b"] = [np.zeros_like(layer.B) for layer in self.network]
        
        self.adam["v_tensor"]["w"] = [np.zeros_like(layer.W) for layer in self.network]
        self.adam["v_tensor"]["b"] = [np.zeros_like(layer.B) for layer in self.network]
        self.adam["inited"] = True

    def set_adam_params(self,**params) -> None :
        for key in ("beta1","beta2","epsilon"):
            if key in params:
                self.adam[key] = params[key]

    def add_layer(self,*layers):
        "Metodo que adiciona camadas da DNN"
        for layer in layers:
            self.network.append(layer)
            
    def __call__(self,Input):
        "Metodo foward da DNN"
        Foward = self.__data_treatment__(Input)
        for layer in self.network:
            Foward = layer(Foward)
        return Foward
    
    "Metodo responsavel por fazer a retropropagação por minibatch e Adam"      
    def fit(self,Xtrain,Ytrain,epochs = 1):
        eta = 1/self.batch_size
        dW_acumulator = [np.zeros_like(layer.W) for layer in self.network]
        dB_acumulator = [np.zeros_like(layer.B) for layer in self.network]
        
        if not self.adam["inited"]:
            self.set_adam_tensors()

        for epoch in range(epochs):
            for x,y in zip(Xtrain,Ytrain):
                predicted = self(x)
                dX = self.loss_function(predicted,y)
                dX = self.__data_treatment__(dX)
                
                for layer_index in range(len(self.network)-1, -1, -1):
                    grad = self.network[layer_index].backpropagation(dX,1)
                    dX,dW,dB = grad
                    dW_acumulator[layer_index] += dW              
                    dB_acumulator[layer_index] += dB
                
            self.adam["t"] += 1

        b1 = self.adam["beta1"]                 
        b2 = self.adam["beta2"]
        epsilon = self.adam["epsilon"]
        t = self.adam["t"]
        
        #Metodo responsavel por atualizar os pesos
        for index,layer in enumerate(self.network):
            #First derivative calculation , W and B
            aux1w = dW_acumulator[index] * (1 - b1) 
            aux1b = dB_acumulator[index] * (1 - b1)
            self.adam["m_tensor"]["w"][index] = (b1 * self.adam["m_tensor"]["w"][index]) + aux1w
            self.adam["m_tensor"]["b"][index] = (b1 * self.adam["m_tensor"]["b"][index]) + aux1b

            #First derivative versor correction
            b1_norm = (1 - b1**t)
            m_hat_w = self.adam["m_tensor"]["w"][index] / b1_norm
            m_hat_b = self.adam["m_tensor"]["b"][index] / b1_norm

            #Second derivative calculation, W and B
            aux2w = (1 - b2) * dW_acumulator[index] ** 2
            aux2b = (1 - b2) * dB_acumulator[index] ** 2
            self.adam["v_tensor"]["w"][index] = (b2 * self.adam["v_tensor"]["w"][index]) + aux2w
            self.adam["v_tensor"]["b"][index] = (b2 * self.adam["v_tensor"]["b"][index]) + aux2b

            #Second derivative versor correction
            b2_norm = (1 - b2**t)
            v_hat_w = self.adam["v_tensor"]["w"][index] / b2_norm
            v_hat_b = self.adam["v_tensor"]["b"][index] / b2_norm
            
            #Updating wheights and bias 
            layer.W -= eta * self.alpha * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
            layer.B -= eta * self.alpha * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

class DDQN(DNN):

    def __init__(self, **parameters):
        self.params_init()
        self.policy = self.linear_epsilon_greedy
        self.gamma = float()
        self.target_network = list()
        self.memory_size = int(2000)
        self.learn_count = int(0)
        self.epsilon = float()
        self.epsilon_decay = float()
        self.epsilon_min = float()
        self.set_parameter(**parameters)
        self.memorias = deque(maxlen=self.memory_size)

    def __call__(self, state, network = None):
        if network == None:
            network = self.network
        Foward = self.__data_treatment__(state)
        for layer in network:
            Foward = layer(Foward)
        return Foward

    def add_layer(self, *layers):
        for layer in layers:
            self.network.append(layer)
            self.target_network.append(layer)

    def act(self, state):
        self.state = self.__data_treatment__(state)
        return self.policy(mode=0)

    def policy_update(self):
        self.policy(mode=1)

    def memorizar(self, *memoria):
        "Função que memoriza a tupla memoria (s,a,r,s´,fim) do agente"
        self.memorias.append(tuple(memoria))
        self.learn_count+=1
        "Executa o treinamento da rede caso tamanho do batch seja alcançado"
        if self.learn_count % self.batch_size == 0:
            self.replay()

    def replay(self):
        "Metodo de replay das memorias"
        train_memorys = rand.sample(self.memorias, self.batch_size)
        Xtrain = []
        Ytrain = []
        for state, action, reward, next_state, end in train_memorys:
            Xtrain.append(state)
            target = self(state, self.network)
            if end:
                target[0][action] = reward
            else:
                next_Q = self(next_state, self.target_network)
                target[0][action] = reward + self.gamma * np.amax(next_Q)
            Ytrain.append(target)
        self.fit(Xtrain,Ytrain,epochs=1)

        if self.learn_count >= self.memory_size:
            self.target_network = copy.deepcopy(self.network)
            self.learn_count = 0


    def linear_epsilon_greedy(self, mode):
        "Modo de ação"
        if mode == 0:
            "Metodo que faz a ação do agente baseado no epsilon_greedy"
            if np.random.rand() <= self.epsilon:
                return rand.randrange(self.action_size)
            else:
                return np.argmax(self(self.state, self.network)[0])
        "modo de atualização"
        if mode == 1:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
    def exp_epsilon_greedy(self, mode):
        "Modo de ação"
        if mode == 0:
            "Metodo que faz a ação do agente baseado no epsilon_greedy"
            if np.random.rand() <= self.epsilon:
                return rand.randrange(self.action_size)
            else:
                return np.argmax(self(self.state, self.network)[0])
        "modo de atualização"
        if mode == 1:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay