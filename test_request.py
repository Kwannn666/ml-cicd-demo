import requests, json

payload = {
  "rows": [
    {"f1":0.1,"f2":1.2,"f3":-0.2,"f4":3.3,"f5":0.5,"f6":1.1},
    {"f1":-0.5,"f2":0.0,"f3":1.2,"f4":-2.1,"f5":0.3,"f6":0.7}
  ]
}
r = requests.post("http://localhost:8000/predict", json=payload)
print(r.status_code, r.json())
