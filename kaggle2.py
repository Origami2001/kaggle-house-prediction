import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import xgboost as xgb
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

test_ids=test_data['Id']

for df in [train_data,test_data]:
    df['Totalarea']=df['TotalBsmtSF']+df['1stFlrSF']+df['2ndFlrSF']+df['GarageArea']

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

train_lables=np.log(train_data['SalePrice'].values)

def log_rmse(y_true,y_pred):
    y_pred=np.maximum(y_pred,1)
    return np.sqrt(mean_absolute_error(np.log(y_true),np.log(y_pred)))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
ridge=Ridge()

param_grid={'alpha': [0.1, 1, 10, 20, 50, 100]}
grid_search=GridSearchCV(ridge,param_grid,scoring="neg_mean_squared_error",cv=kf)
grid_search.fit(train_features,train_lables)
best_alpha=grid_search.best_params_['alpha']
print("最佳alpha：",best_alpha)

ridge_best=Ridge(alpha=best_alpha)
neg_mse_scores=cross_val_score(ridge_best,train_features,train_lables,scoring="neg_mean_squared_error",cv=kf)
rmse_scores=np.sqrt(-neg_mse_scores)
print("Ridge交叉验证的log RMSE：",rmse_scores.mean())

ridge_best.fit(train_features,train_lables)
test_pred_log=ridge_best.predict(test_features)
max_log_price=np.log(train_data['SalePrice'].max())
test_pred_log=np.clip(test_pred_log,None,max_log_price)
test_pred=np.exp(test_pred_log)
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_pred
})
submission.to_csv("sub.csv", index=False)
print("提交文件已保存！")