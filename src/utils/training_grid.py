from itertools import product
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
import numpy as np, torch
class GridNet(NeuralNetRegressor):
    def __init__(self,*a,l1_lambda=0.0,**kw):super().__init__(*a,**kw);self.l1_lambda=l1_lambda
    def get_loss(self,y_pred,y_true,*_,**__):
        loss=super().get_loss(y_pred,y_true);l1=sum(p.abs().sum() for p in self.module_.parameters());return loss+self.l1_lambda*l1/len(y_true)

def train_grid(mdl,X_tr,y_tr,X_val,y_val,space,epochs, device="cpu"):
    best,best_cfg,best_mse=None,None,float("inf")
    keys,vals=list(space.keys()),list(space.values())
    for combo in product(*vals):
        hp=dict(zip(keys,combo))
        opt_name=hp.pop("optimizer","Adam")
        opt=getattr(torch.optim,opt_name)
        lr=hp.pop("lr",1e-3)
        wd=hp.pop("weight_decay",0.0)
        l1=hp.pop("l1_lambda",0.0)
        current_batch_size = hp.pop("batch_size", 256)

        module_specific_params = {f"module__{k}": v for k, v in hp.items()}

        net=GridNet(
            module=mdl,
            module__n_feature=X_tr.shape[1],
            module__n_output=1,
            **module_specific_params,
            max_epochs=epochs,
            batch_size=current_batch_size,
            optimizer=opt,
            lr=lr,
            optimizer__weight_decay=wd,
            l1_lambda=l1,
            iterator_train__shuffle=True,
            callbacks=[EarlyStopping(patience=10)],
            device=device
            )
        net.fit(X_tr, y_tr)
        mse=mean_squared_error(y_val, net.predict(X_val))
        if mse<best_mse:
            best_hp_for_return = hp.copy()
            best_hp_for_return.update({
                "optimizer": opt_name, "lr": lr, "weight_decay": wd,
                "l1_lambda": l1, "batch_size": current_batch_size
            })
            best, best_mse, best_cfg = net, mse, best_hp_for_return
    return best,best_cfg,best_mse