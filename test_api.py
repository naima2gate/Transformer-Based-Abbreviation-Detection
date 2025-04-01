import requests

input_text = "EPI = ECHO PLANAR IMAGING"
url = "http://127.0.0.0:5001/predict"
data = {"text": input_text}
response = requests.post(url, json=data)

if response.status_code == 200:
    predictions = response.json()["predictions"]
    print("Predicted labels:", predictions)
else:
    print("Error:", response.text)