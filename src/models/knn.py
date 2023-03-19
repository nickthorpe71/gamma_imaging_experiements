from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def run_knn(df_X_train: DataFrame, df_y_train: DataFrame, df_X_test: DataFrame, df_y_test: DataFrame) -> None:
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(df_X_train, df_y_train)
    df_y_pred = knn_model.predict(df_X_test)
    print(classification_report(df_y_test, df_y_pred))
    
