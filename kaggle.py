import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

all_features=pd.concat((train_data.drop(['Id','SalePrice'],axis=1),test_data.drop(['Id'],axis=1)),axis=0)

numeric_features=all_features.select_dtypes(include=[np.number]).columns

all_features[numeric_features]=all_features[numeric_features].apply(
    lambda x:(x-x.mean())/(x.std())
)
all_features[numeric_features]=all_features[numeric_features].fillna(0)

all_features=pd.get_dummies(all_features,dummy_na=True)

n_train=train_data.shape[0]
train_features=all_features[:n_train].values
test_features=all_features[n_train:].values

train_lables=train_data['SalePrice'].values

def log_rmse(y_true,y_pred):
    y_pred=np.maximum(y_pred,1)
    return np.sqrt(mean_absolute_error(np.log(y_true),np.log(y_pred)))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

neg_mse_scores = cross_val_score(model, train_features, train_lables, scoring="neg_mean_squared_error", cv=kf)
rmse_scores = np.sqrt(-neg_mse_scores)
print("交叉验证的 log RMSE：", rmse_scores.mean())

model.fit(train_features,train_lables)
test_pred=model.predict(test_features)
test_ids = pd.read_csv('test.csv')['Id']
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_pred
})

submission.to_csv("submission.csv", index=False)
print("提交文件已保存！")