from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

#https://docs.sqlalchemy.org/en/20/orm/

db=SQLAlchemy()
migrate=Migrate()

def create_app():
    app = Flask(__name__)
    print(f'__name__ : {__name__}')
    

    from .views import main_views
    app.register_blueprint(main_views.bp)
    
    return app