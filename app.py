from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
from shapely.geometry import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from flask import Flask, request, jsonify
from shapely.geometry import Polygon, LineString, box, MultiPolygon
from shapely.ops import unary_union
import numpy as np
from scipy.spatial import Voronoi


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 1)  # Ensure image is read

    img = remove_white_background(img)
    _, buf = cv2.imencode('.png', img)
    buffer = io.BytesIO(buf)
    return send_file(buffer, mimetype='image/png')

def remove_white_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    return dst


@app.route('/generate-rooms', methods=['POST'])
def generate_rooms():
    data = request.get_json()
    if 'polygon' not in data or 'roomPoints' not in data or 'roomLines' not in data:
        return jsonify({'error': 'Missing required data'}), 400

    polygon_points = [(p['x'], p['y']) for p in data['polygon']]
    room_points = [(p['x'], p['y']) for p in data['roomPoints']]
    room_lines = data['roomLines']
    polygon = Polygon(polygon_points)

    # Handle room lines first to create fixed rooms
    rooms, new_boundaries = handle_room_lines(room_lines, polygon)

    # Update polygon to exclude areas covered by line-based rooms
    if new_boundaries:
        polygon = polygon.difference(unary_union(new_boundaries))

    # Generate Voronoi cells within the updated polygon
    if room_points:
        voronoi = Voronoi(np.array(room_points))
        rooms.extend(generate_voronoi_rooms(voronoi, polygon))

    return jsonify(rooms), 200

def handle_room_lines(room_lines, polygon):
    rooms = []
    new_boundaries = []
    for line in room_lines:
        path = LineString([(line['x1'], line['y1']), (line['x2'], line['y2'])])
        buffer_width = 10  # Adjust width for corridor
        corridor = path.buffer(buffer_width, cap_style=2)
        intersected_corridor = corridor.intersection(polygon)
        new_boundaries.append(intersected_corridor)
        rooms.extend(create_room_data(intersected_corridor))
    return rooms, new_boundaries

def create_room_data(geometry):
    """ Helper function to create room data from geometries. """
    rooms = []
    if isinstance(geometry, Polygon):
        rooms.append({
            'type': 'polygon',
            'points': [{'x': point[0], 'y': point[1]} for point in geometry.exterior.coords]
        })
    elif isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            rooms.append({
                'type': 'polygon',
                'points': [{'x': point[0], 'y': point[1]} for point in poly.exterior.coords]
            })
    return rooms

def generate_voronoi_rooms(voronoi, polygon):
    regions, vertices = voronoi.regions, voronoi.vertices
    rooms = []
    for point_idx, region_idx in enumerate(voronoi.point_region):
        region = regions[region_idx]
        if -1 not in region:
            poly_points = vertices[region]
            poly = Polygon(poly_points).intersection(polygon)
            rooms.extend(create_room_data(poly))
    return rooms

if __name__ == '__main__':
    app.run(debug=True)