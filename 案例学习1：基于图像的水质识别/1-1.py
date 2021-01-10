from data_process import get_img_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report, confusion_matrix

data, labels = get_img_data()

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)
Dtc = DecisionTreeClassifier().fit(data_train, labels_train)
pre = Dtc.predict(data_test)

confusion_matrix(labels_test, pre)
classification_report(labels_test, pre)

