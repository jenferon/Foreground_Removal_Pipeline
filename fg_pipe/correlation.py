import numpy as np

def pearson_correl(x,y):
    """_summary_

    Args:
        x (array, float): _description_
        y (array, float): _description_

    Returns:
        _type_: _description_
    """
 return (np.sum((x-np.mean(x))*(y-np.mean(y))))/(np.sqrt(np.sum((x-np.mean(x))**2)*np.sum((y-np.mean(y))**2)))