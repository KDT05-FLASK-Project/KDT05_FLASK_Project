from flask import Blueprint, render_template, request 
from datetime import datetime

bp = Blueprint('NA', __name__, template_folder = 'templates',
                    url_prefix="/na_db")

@bp.route('/')
def index():
    return f"명노아의 페이지입니다!"
    