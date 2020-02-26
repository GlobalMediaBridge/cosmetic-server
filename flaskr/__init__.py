import os
import platform
from flask import Flask, flash, send_file, render_template, redirect, request, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np

UPLOAD_FOLDER = 'static/images'
if platform.system() == 'Windows':
    UPLOAD_FOLDER = 'static\\images'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, instance_relative_config=True,
            static_url_path='/static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def inputTest():
    return render_template('main.html')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'fail'

        f = request.files['image']

        if f.filename == '':
            return 'fail'

        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            f.save(os.path.join(current_dir,
                                app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('inputTest'))

        return 'fail'


@app.route('/images')
def show_images_list():
    current_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), UPLOAD_FOLDER)
    file_list = map(lambda x: '<li>%s</li>' % x, os.listdir(current_dir))
    return ''.join(file_list)


@app.route('/images/<filename>')
def show_images(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return send_file(os.path.join(current_dir,
                                  app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')


@app.route('/magic', methods=['GET', 'POST'])
def magic_wand():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'fail'

        f = request.files['image']
        x = int(request.form['x'])
        y = int(request.form['y'])

        if f.filename == '':
            return 'fail'

        img = cv2.imdecode(np.fromstring(
            f.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        flood_mask = getFloodMask(img, x, y)
        mean = getMean(img, flood_mask)
        return '(%s)' % ', '.join(map(str, mean))


def getMean(img, mask):
    mean = cv2.mean(img, mask)
    return mean


def getFloodMask(img, x, y, tolerance=32):
    h, w = img.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_fill_flags = (
        4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8
    )
    tolerance = (tolerance,) * 3
    flood_mask[:] = 0
    cv2.floodFill(
        img,
        flood_mask,
        (x, y),
        0,
        tolerance,
        tolerance,
        flood_fill_flags,
    )

    return flood_mask[1:-1, 1:-1].copy()
