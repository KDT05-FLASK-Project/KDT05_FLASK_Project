from flask import Blueprint, render_template, request 
from datetime import datetime

bp = Blueprint('SM', __name__, template_folder = 'templates',
                    url_prefix="/sm_db")

@bp.route('/')
def index():
    return render_template()

