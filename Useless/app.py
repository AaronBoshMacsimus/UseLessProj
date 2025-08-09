from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

app = Flask(
    __name__,
    static_folder="frontend/dist",    # This is where Vite build goes
    static_url_path=""
)
CORS(app)

# In-memory layout data
image_placements = [
    {"filename": "img1.png", "x": 100, "y": 50},
    {"filename": "img2.png", "x": 250, "y": 150}
]

# GET layout
@app.route('/api/layout')
def layout():
    return jsonify({"image_placements": image_placements})

# POST to add image
@app.route('/api/add', methods=['POST'])
def add():
    data = request.json
    image_placements.append(data)
    return jsonify({"status": "added"})

# POST to remove image
@app.route('/api/remove', methods=['POST'])
def remove():
    data = request.json
    global image_placements
    image_placements = [img for img in image_placements if img["filename"] != data["filename"]]

    return jsonify({"status": "removed"})


@app.route('/api/move', methods=['POST'])
def move():
    data = request.json
    filename = data.get("filename")
    new_x = data.get("x")
    new_y = data.get("y")

    for img in image_placements:
        if img["filename"] == filename:
            img["x"] = new_x
            img["y"] = new_y
            return jsonify({"status": "moved"})

    return jsonify({"error": "image not found"}), 404

# Serve images from dist/models/
@app.route('/models/<filename>')
def serve_model_image(filename):
    return send_from_directory("frontend/dist/models", filename)

# Serve React app and assets
@app.route('/')
@app.route('/<path:path>')
def serve_react(path=''):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
