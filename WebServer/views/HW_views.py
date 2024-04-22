from flask import Blueprint, render_template, request 
from datetime import datetime
import torch
from transformers import BertTokenizer
from torch import optim
from transformers import BertForSequenceClassification
from WebServer import db
from WebServer.models import HW_DB

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path="bert-base-multilingual-cased", num_labels=2
).to(device)
model_state_dict = torch.load('./양현우/BERTSequenceClassification.pt', map_location=device)
model.load_state_dict(model_state_dict)
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-multilingual-cased", do_lower_case=False
)

# 긍정부정판별 함수
def plusminus(test_text):
    tokenized = tokenizer(text=[test_text],padding="longest",truncation=True, return_tensors='pt')
    resultList=torch.sigmoid(model(tokenized['input_ids']).logits).squeeze().tolist()
    result = resultList.index(max(resultList))
    return result, max(resultList)





bp = Blueprint('HW', __name__, template_folder = 'templates',
                    url_prefix="/hw_db")


@bp.route('/')
def index():
    return render_template("HW/index.html")

@bp.route('/result', methods=['POST'])
def result():
    req_dict = request.form.to_dict()
    text = req_dict.get('story')
    a,b = plusminus(text)
    q = HW_DB(input=text, output=f'{"긍정적 확률 :" if a else "부정적 확률 :"}{b}', create_date=datetime.now())  
    db.session.add(q)
    db.session.commit()
    table_list = HW_DB.query.order_by(HW_DB.create_date.desc())

    return render_template("HW/result.html", value=f'{"긍정적 확률 :" if a else "부정적 확률 :"}{b}',
                           table_list=table_list)
