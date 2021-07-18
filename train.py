from pandas.core.indexing import maybe_convert_ix
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

files = os.listdir()

file_name = "train_data.csv"

Data = pd.read_csv(file_name).values
np.random.shuffle(Data)
print(f'Number of Users = {max(Data[:,1])}')
print(f'Number of Movies = {max(Data[:,0])}')
num_users = max(Data[:,1])+1
num_movies = max(Data[:,0])+1
#mat_data = np.zeros((num_movies,num_users))

'''for entry in Data:
    user = entry[1]
    movie =  entry[0]
    mat_data[movie,user] = entry[2]'''

#print(mat_data[:10,:10])

K = [100,500,1000,1500]
#K = [2000,2500,3000,3500,4000,4500]
k_fold_results = dict()

for k in K:
    print(f'Epochs with k = {k}')
    U = np.maximum(0,np.random.rand(num_movies,k)/100)
    V = np.maximum(0,np.random.rand(k,num_users)/100)

    num_epochs = 75
    lr = 0.002
    train_loss = []
    valid_loss= []

    N = Data.shape[0]
    split = int(0.75*N)
    train_set,valid_set = Data[:split], Data[split:]

    epoch = 0
    while epoch < num_epochs:
        t_loss = 0
        for entry in train_set:
            row = entry[0]
            col = entry[1]
            rating = entry[2]
            
            temp = (rating - U[row,:]@V[:,col])
            #print(f"temp:{temp}")
            grad_u = temp*V[:,col]
            grad_v = temp*U[row,:]

            U[row,:] += lr*grad_u
            V[:,col] += lr*grad_v

            t_loss += (rating - U[row,:]@V[:,col])**2

        train_loss.append(t_loss/split)

        v_loss = 0
        for entry in valid_set:
            row = entry[0]
            col = entry[1]
            rating = entry[2]
            v_loss += (rating - U[row,:]@V[:,col])**2
        valid_loss.append(v_loss/(N-split))

        print("Epoch::"+str(epoch)+'\nTraining Loss::'+str(t_loss)+'\tValidation Loss::'+str(v_loss))
        epoch+=1

    plt.title(f"Training vs validation Loss with K = {k}")
    plt.plot(train_loss,color = 'red',label = 'Train_loss')
    #plt.show()

    #plt.title("Valid Loss")
    plt.plot(valid_loss,color = 'blue',label = 'valid_loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig(f"Train_loss vs Val_loss for K = {k}.png")
    plt.close()

    k_fold_results[str(k)] = {'U':U.tolist(),'V':V.tolist(),'validation':valid_set.tolist()}

## end of the k-fold cross validation loop

#data_dif = []

'''for entry in Data:
    row,col,rating = entry[0],entry[1],entry[2]

    print(rating,"pred: ",U[row,:]@V[:,col])
    data_dif.append(abs(rating - max(0,U[row,:]@V[:,col])))
plt.title("Difference Between Actual Data and LMF predicted Data")
plt.plot(np.arange(len(data_dif)),data_dif)
plt.show()'''


with open("Store_k_fold_results.txt",'w') as f:
    json.dump(k_fold_results,f)



