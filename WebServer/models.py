from WebServer import db

    
class NA_DB(db.Model): # 명노아의 데이터베이스 
    id = db.Column(db.Integer, primary_key=True)
    input = db.Column(db.String(200), nullable=False)
    output = db.Column(db.String(200), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)
    
class JT_DB(db.Model): # 변주영의 데이터베이스 
    id = db.Column(db.Integer, primary_key=True)
    input = db.Column(db.String(1000), nullable=False)
    output = db.Column(db.String(200), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)
    
class SM_DB(db.Model): # 이시명의 데이터베이스 
    id = db.Column(db.Integer, primary_key=True)
    input = db.Column(db.String(200), nullable=False)
    output = db.Column(db.String(200), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)
    
class HW_DB(db.Model): # 양현우의 데이터베이스 
    id = db.Column(db.Integer, primary_key=True)
    input = db.Column(db.String(200), nullable=False)
    output = db.Column(db.String(200), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)
