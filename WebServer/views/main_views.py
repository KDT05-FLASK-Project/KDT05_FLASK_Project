from flask import Blueprint, render_template, request 
from datetime import datetime

bp = Blueprint('data', __name__, template_folder = 'templates',
                    url_prefix="/")

@bp.route('/', methods=['GET','POST'])
def index():
    return "hi"
    