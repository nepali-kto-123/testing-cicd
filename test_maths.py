import pandas as pd
import numpy as np
import pickle

with open("./model.pkl", "rb") as f:
    model = pickle.load(f)

print("♨♨ model has been loaded.")    

def test_shape():
    X = np.random.rand(5,1)
    y_pred = model.predict(pd.DataFrame(X, columns = ['x']))

    assert y_pred.shape == (5, )

''' this @ is used when you want to give the like numbers like batch size before hand will work in a loop
@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])  
def test_model_with_various_batch_sizes(model, batch_size, X):
    y_pred = model.predict(X[:batch_size])
    assert len(y_pred) == batch_size

'''

#------------WITH FASTAPI STUFF---------------------------------#

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert isinstance(data["reply"], str)


# run through "pytest -v"    