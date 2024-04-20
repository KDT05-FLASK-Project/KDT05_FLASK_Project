from flask import Blueprint, render_template, request 
from datetime import datetime

bp = Blueprint('JY', __name__, template_folder = 'templates',
                    url_prefix="/jy_db")

@bp.route('/')
def index():
    return f"변주영의 페이지입니다!"
    