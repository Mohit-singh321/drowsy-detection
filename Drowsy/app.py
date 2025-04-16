# app.py
from flask import Flask, render_template, redirect, url_for
from threading import Thread
from drowsiness import run_drowsiness_detection

app = Flask(__name__)
detection_thread = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start')
def start_detection():
    global detection_thread
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = Thread(target=run_drowsiness_detection)
        detection_thread.start()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
