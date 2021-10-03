from typing import Dict, List
import pandas as pd
from scipy.spatial.distance import correlation


def similarty_dict(x: Dict, y: List[Dict], metric=correlation):
    """Calculate dictance of one vector in dict format to other dictinary in list of dict"""
    vecs = pd.DataFrame([x] + y).fillna(0).values
    return [1 - metric(vecs[0], v) for v in vecs[1:]]


