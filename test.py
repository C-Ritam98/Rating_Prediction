import json
import matplotlib.pyplot as plt
import numpy as np

# K = [50,100,200,250,400,500,750,1000,1200,1500,1750,2000,2500]
K = [100,500,1000,1500]

with open('/home/ritam/Desktop/Courses/movie_rating_data/Store_k_fold_results.txt','r') as json_file:
    record = json.load(json_file)
    print("Loaded!")

diff_2 = []
diff_1 = []
diff_0 = []

for k in record.keys():
    print(f'For K = {k}')
    data = record[k]
    U = np.array(data['U'])
    V = np.array(data['V'])
    valid_set = np.array(data['validation'])

    pred = []
    for entry in valid_set:
        row,col,rating = entry[0],entry[1],entry[2]
        temp = round(U[row,:]@V[:,col])
        pred.append(temp)

        #print(f"User {col} rates movie {row}: {rating} while our prediction = {temp}")

    x = np.arange(100)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    ax.bar(x,pred[:100],color='r',width = 0.5)
    ax.bar(x+0.5,valid_set[:100,2],color='b',width = 0.5)
    ax.legend(labels = ["Predicted","Actual"])
    plt.show()

    y = np.array(pred)

    print(f'Max 2 rates apart:{np.mean(abs(y- valid_set[:,2]) <= 2)*100}%')
    print(f'Max 1 rate apart:{np.mean(abs(y- valid_set[:,2]) <= 1)*100}%')
    print(f'Exact matches:{np.mean(abs(y- valid_set[:,2]) == 0)*100}%')
    diff_2.append(np.mean(abs(y- valid_set[:,2]) <= 2)*100)
    diff_1.append(np.mean(abs(y- valid_set[:,2]) <= 1)*100)
    diff_0.append(np.mean(abs(y- valid_set[:,2]) == 0)*100)

plt.plot(K,diff_2,label = '#reviews with max diff 2 from predicted')
plt.plot(K,diff_1,label = '#reviews with max diff 1 from predicted')
plt.plot(K,diff_0,label = '#reviews with exact match with predicted')
plt.legend()
plt.savefig('Accuracy_with_change_in_k.png')
plt.close()