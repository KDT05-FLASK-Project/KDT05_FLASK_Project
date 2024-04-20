from flask import Blueprint, render_template, request 
from datetime import datetime

bp = Blueprint('HW', __name__, template_folder = 'templates',
                    url_prefix="/hw_db")

@bp.route('/')
def index():
    return f"양현우의 페이지입니다!"
    