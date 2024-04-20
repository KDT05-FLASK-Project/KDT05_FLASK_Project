from flask import Blueprint, render_template, request 
from datetime import datetime

bp = Blueprint('data', __name__, template_folder = 'templates',
                    url_prefix="/sm_db")

@bp.route('/')
def index():
    return f"이시명의 페이지입니다!"
    