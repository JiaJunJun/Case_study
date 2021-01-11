from data_process import get_img_data
from sklearn.neural_network import MLPClassifier

tr_path = 'number_images/trainImages/'
te_path = 'number_images/testImages/'

data_tr, labels_tr = get_img_data(tr_path)
data_te, labels_te = get_img_data(te_path)

Mlp = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=1000)
Mlp.fit(data_tr, labels_tr)
score = Mlp.score(data_te, labels_te)
print(score)