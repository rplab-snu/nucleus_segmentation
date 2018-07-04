import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_similarity_score, confusion_matrix


target = np.random.randint(2, size=(448, 448))
output = np.random.randint(2, size=(448, 448))

target = np.array([[0,0,0],[0,0,0],[0,0,0]]).flatten()
output = np.array([[0,0,0],[1,1,0],[0,0,0]]).flatten()
print(target, output)

# a = jaccard_similarity_score(target, output)
b = [0,0,0,0]
print(b)
b += confusion_matrix(target, output).ravel()
print(b)
b += confusion_matrix(target, output).ravel()
print(b)
b += confusion_matrix(target, output).ravel()
# a = recall_score(target, output)
# a = confusion_matrix(target, output).ravel()
print(b)
