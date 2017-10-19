# README Inference

This is a simple API based on flask and our model

To start the API:
```
python app.py
```

Now in another terminal send the api an image:
```
 (echo -n '{"data": "'; base64 examples/retail.jpeg; echo '"}') | curl -X POST -H "Content-Type: application/json" -d @- http://0.0.0.0:8080
 (echo -n '{"data": "'; base64 examples/other.jpeg; echo '"}') | curl -X POST -H "Content-Type: application/json" -d @- http://0.0.0.0:8080
```

or for convenience, do the following

1. source this function:
```
. utils/fc
```

2. now just supply filename
```
feed examples/retail.jpeg
```

should result in the following:
```
{
  "predictions": [
    {
      "description": "retail", 
      "label": 1, 
      "probability": 97.92
    }
  ]
}
```
