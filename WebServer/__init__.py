from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db=SQLAlchemy()
migrate=Migrate()

def create_app():
    app = Flask(__name__)
    print(f'__name__ : {__name__}')
    
    app.config.from_pyfile('config.py') # 설정 내용 로딩 
    
    # ORM
    db.init_app(app)
    migrate.init_app(app, db)
    
    # 테이블 클래스 
    from . import models
    
    from .views import main_views, NA_views, HW_views, SM_views, JY_views
    
    app.register_blueprint(main_views.bp)
    app.register_blueprint(NA_views.bp)
    app.register_blueprint(HW_views.bp)
    app.register_blueprint(SM_views.bp)
    app.register_blueprint(JY_views.bp)
    
    
    return app