import model.ChakeList as mo
import json
data = {'sex':1,'age':2,'alchol':1.1}

def test(data):
    print(data['sex']+data['age'])
    return data

test(data)
