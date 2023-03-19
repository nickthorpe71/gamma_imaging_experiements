from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier

def knn(df_X_train: DataFrame, df_y_train: DataFrame, df_X_test: DataFrame) -> DataFrame:
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(df_X_train, df_y_train)
    df_y_pred = knn_model.predict(df_X_test)
    
    return df_y_pred
    
