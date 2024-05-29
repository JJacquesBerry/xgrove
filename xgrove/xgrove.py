import sklearn
import sklearn.tree
def test():
    print("test successful")

def xgrove(
        model,
        data,
        ntrees = [4, 8, 16, 32, 64, 128],
        pfun = NULL,
        shrink = 1,
        #b.frac = 1,
        seed = 42):
    
    sklearn.tree()