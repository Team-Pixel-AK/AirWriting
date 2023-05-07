from flask import Flask, render_template, request, url_for, redirect
import subprocess

app = Flask(__name__)


@app.route('/')
def load_website():
    return render_template('index.html')


@app.route('/static/style.css')
def serve_style():
    return app.send_static_file('style.css')


@app.route('/static/myvideo.mp4')
def serve_video():
    return app.send_static_file('myvideo.mp4')


@app.route('/execute1', methods=['POST'])
def execute1():
    if request.method == 'POST':
        subprocess.call(['python3', 'video.py'])
        return render_template("index.html")


@app.route('/execute2', methods=['POST'])
def execute2():
    if request.method == 'POST':
        subprocess.call(['python3', 'video.py'])
        return render_template("index.html")


@app.route('/execute3', methods=['POST'])
def execute3():
    if request.method == 'POST':
        subprocess.call(['python3', 'video.py'])
        return render_template("index.html")


@app.route('/execute4', methods=['POST'])
def execute4():
    if request.method == 'POST':
        subprocess.call(['python3', 'video.py'])
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
