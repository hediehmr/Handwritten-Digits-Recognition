import keras
from keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:1000,:,:]
X_test = X_test[:1000,:,:]
y_train = y_train[:1000]
y_test  = y_test[:1000]

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Each image has Intensity from 0 to 255
#so convert it to range 0 to 1
X_train = X_train/255 
X_test = X_test/255
# X_train = np.array(X_train)
X_train = X_train.astype(int)
X_test= X_test.astype(int)
#Because of the matrix multipication we must change the shape of the array.
#(AxB * BxC = AxC) So we will shape the 28x28 array to 1x784.
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0],num_pixels)
X_test = X_test.reshape(X_test.shape[0],num_pixels)
W_firtlayyer=np.random.random((784,10))
W_seclayyer=np.random.random((10,10))
net_hidden_nodes=np.zeros(10)
net_out_nodes=np.zeros(10)
Output_H=np.zeros(10)
Output_O=np.zeros(10)
sigma=0
sig_out=np.zeros(10)
sig_Hiden=np.zeros(10)

#algorithm
for i1 in range(0,2000):# تعداد دفعات تکرار الگوریتم
    for num_input_data in range (0,1000):#تعداد دفعات تکرار به ازای داده ها

        net_hidden_nodes = np.matmul(X_train[num_input_data,:],W_firtlayyer)#ok
        Output_H=1/(1+(np.exp(-net_hidden_nodes)))#ok
        net_out_nodes = np.matmul(Output_H,W_seclayyer)#calculate output nodes' net
        a=np.exp(-net_out_nodes)
        Output_O=1 / (1 + a) 
        a=np.multiply(Output_O,(1-Output_O))#ok
        b=np.subtract(y_train[num_input_data,:],Output_O)
        sig_out=np.multiply(a,b)#ok
        sigma += np.matmul(sig_out,W_seclayyer.T)
        a=np.multiply(Output_H,(1-Output_H))#ok
        sig_Hiden=sigma*a
        sigma=0#ok
        X_train_i=np.array(X_train[num_input_data,:])[np.newaxis]
        X_train_T=X_train_i.T
        b=np.array(sig_Hiden)[np.newaxis]
        W_firtlayyer += np.matmul(X_train_T,b)
        a=np.array(Output_H)[np.newaxis]
        W_seclayyer +=  np.matmul(a,sig_out)


for cnt in range (0,5):
    net_hidden_nodes = np.matmul(X_test[cnt,:],W_firtlayyer)#calculate hidden 
    Output_H=1/(1+(np.exp(-net_hidden_nodes)))           
    net_out_nodes = np.matmul(Output_H,W_seclayyer)
    Output_O=1/(1+(np.exp(-net_out_nodes)))
    print (str(y_test[cnt]))
    print (str(Output_O))
