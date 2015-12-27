## ELM

### Simple implementation of Extreme Learning Machine

Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Siew
Extreme Learning Machine: A New Learning Scheme of Feedforward Neural Networks

Example of usage:

    from elm import Elm
    ...
    clf = Elm(100, 0.01, 42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


