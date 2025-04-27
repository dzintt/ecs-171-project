from flask import Flask, render_template_string
import matplotlib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('./index.html')