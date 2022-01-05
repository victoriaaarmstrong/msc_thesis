from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

import time


def SVM(X, Y):
    """
    :param X:
    :param Y:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.70, test_size=0.30,
                                                        random_state=101)

    start = time.process_time()

    classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1, multi_class='ovr',
                           fit_intercept=True, intercept_scaling=1, class_weight='balanced', verbose=0,
                           random_state=101)

    classifier = classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)

    print(time.process_time() - start)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    return

