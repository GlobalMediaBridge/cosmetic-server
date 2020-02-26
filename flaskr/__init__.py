import os
import platform
from flask import Flask, flash, send_file, render_template, redirect, request, url_for
from werkzeug.utils import secure_filename

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
