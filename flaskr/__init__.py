from flask import Flask, render_template

app = Flask(__name__, instance_relative_config=True,
            static_url_path='/static')


@app.route('/')
def inputTest():
    return render_template('main.html')
