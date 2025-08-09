import requests

# Move image named 'img1.png' to new coordinates
requests.post("http://localhost:5000/api/move", json={
    "filename": "img1.png",
    "x": 300,
    "y": 150
})