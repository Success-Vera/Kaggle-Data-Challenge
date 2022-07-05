import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from models import KernelRidgeRegression

# """### Data Collection"""

"""
Uncomment the following lines to download the data. But if you have the data, you can place your path as you run the script.
"""
# import wget
# Xte = 'https://raw.githubusercontent.com/Success-Vera/Kaggle-Data-Challenge/main/data/Xte_vectors.csv'
# Xtr='https://raw.githubusercontent.com/Success-Vera/Kaggle-Data-Challenge/main/data/Xtr_vectors.csv'
# Ytr='https://raw.githubusercontent.com/Success-Vera/Kaggle-Data-Challenge/main/data/Ytr.csv'
# # X_tr_vec = wget.download(Xtr)
# # X_te_vec = wget.download(Xte)
# # Y_tr_vec=wget.download(Ytr)


def main():
    h=input("enter the training data path:")
    m=input("enter the targets path:")
    n=input("enter the submission file path:")
    print("----------Processing---------------\n Your results would appear in some few seconds!")  
    x=pd.read_csv(h)
    y=pd.read_csv(m)
    sub=pd.read_csv(n)

    x.drop(['Id'],axis=1,inplace=True)
    y.drop(['Id'],axis=1,inplace=True)
    y['Covid'].replace(0,-1,inplace=True)
    corr = np.abs(x.corr())
    upper_tri = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
    drop_ = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
    x.drop(drop_,axis=1,inplace=True)
    dataset=pd.concat([x,y["Covid"]],axis=1)

    fold_score=[]
    sub_fold_pred=[]

    X=dataset
    n_folds=5
    n=X.shape[0]
    X_train=X.drop(["Covid"],axis=1)
    sub.drop(drop_,axis=1,inplace=True)
    Y_train=y
    fold_size=len(X)//n_folds
    kernel = 'rbf'
    lambd = 0.001
    sigma = 0.1
    np.random.seed(2022)


    model = KernelRidgeRegression(
        kernel=kernel,
        lambd=lambd,
        sigma=sigma
        )
    sample_weights = np.random.rand(len(X_train)-400)
    for folds in range(n_folds):
        indices=np.arange(folds*fold_size,(1+folds)*fold_size, dtype=int)
        
        x_val=X_train.iloc[indices]
        y_val=Y_train.iloc[indices]
        x_train=X_train[~X_train.apply(tuple,1).isin(x_val.apply(tuple,1))]
        y_train=Y_train.iloc[x_train.index]

        sample_weights = np.random.rand(x_train.shape[0])
        model.fit(x_train.to_numpy(), y_train.to_numpy(), sample_weights=sample_weights)
        ypred = np.sign(np.mean(model.predict(x_val.to_numpy()),axis=1))
        fold_score.append(f1_score(y_val,ypred))
        sub_pred=model.predict(sub.iloc[:,1:].to_numpy())
        sub_fold_pred.append(sub_pred)

    submit=np.sign(np.mean(np.mean(sub_fold_pred,axis=0),axis=1)).astype(int) #we take the mean of the predictions from each fold. Then take the sign of it
    sub_pred=pd.DataFrame(submit,columns=['Covid'])
    ss_pred=pd.concat([sub['Id'],sub_pred],axis=1)
    ss_pred['Covid'].replace(-1,0,inplace=True)
    sample_sub=ss_pred.to_csv('8_sub.csv',index=False)
    print("--------------------Completed!------------------------------")
    return sample_sub


if __name__=="__main__":
    main()