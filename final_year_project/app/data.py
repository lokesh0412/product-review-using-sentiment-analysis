from flask import Flask,render_template,request
import requests,json
app = Flask(__name__)
@app.route('/')
