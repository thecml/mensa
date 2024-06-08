from data_loader import (LinearSyntheticDataLoader,
                         NonlinearSyntheticDataLoader,
                         CompetingRiskSyntheticDataLoader,
                         MultiEventSyntheticDataLoader)

if __name__ == '__main__':
    dl = LinearSyntheticDataLoader().load_data()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dl.split_data(train_size=0.7,
                                                                             valid_size=0.5)
    print(X_train.shape)
    
    dl = NonlinearSyntheticDataLoader().load_data()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dl.split_data(train_size=0.7,
                                                                             valid_size=0.5)
    print(X_train.shape)
    
    dl = CompetingRiskSyntheticDataLoader().load_data()
    (train_pkg, valid_pkg, test_pkg) = dl.split_data(train_size=0.7, valid_size=0.5)
    print(train_pkg[0].shape)
    
    dl = MultiEventSyntheticDataLoader().load_data()
    (train_pkg, valid_pkg, test_pkg) = dl.split_data(train_size=0.7, valid_size=0.5)
    print(train_pkg[0].shape)