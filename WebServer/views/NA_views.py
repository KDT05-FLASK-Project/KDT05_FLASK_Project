from flask import Blueprint, render_template, request, flash, redirect, url_for
from datetime import datetime
from WebServer.models import NA_DB
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch 
from gensim.models import Word2Vec
import re
from WebServer import db
import sys
from WebServer.templates.NA.Youtube_Channel import get_channel_info


tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
model = torch.load("명노아/model/model100.pth", map_location=torch.device('cpu'))
model.eval()

# 댓글 생성 함수 선언
def generate_text(text):
    
    prompt_text = text
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    output = model.generate(input_ids=input_ids,max_length=100,num_return_sequences=1,temperature=0.7, top_k=50, pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,bos_token_id=tokenizer.bos_token_id,num_beams=2,early_stopping=True,do_sample=True,
    )
    for i, sequence in enumerate(output):
        result = tokenizer.decode(sequence, skip_special_tokens=True)
    return result


bp = Blueprint('NA', __name__, template_folder = 'templates',
                    url_prefix="/na_db")


#======================================================================================
@bp.route('/')
def index():
    return render_template("NA/index.html")

@bp.route('/result', methods=['GET','POST'])
def show_result():
    
    req_dict = request.form.to_dict() # 값들 갖고 오기 
    text = req_dict.get('name')
    simir_text = req_dict.get('simir')
    
    if text : # 댓글 생성 함수======================================================1-1
        result = generate_text(text)
        # 알파벳과 특수기호를 제거하는 정규표현식
        pattern = r'[^가-힣\s.,!?]'
        # 각 내용에 대해 정규표현식을 적용하여 필터링
        result = re.sub(pattern, '', result)

    if simir_text: # 유사도 함수====================================================1-2
        word2vec = Word2Vec.load("명노아/model/Gensim.model")
        result2 = word2vec.wv.most_similar(simir_text, topn=5)

    # 테이블 값 갖고오기 
    table_list = NA_DB.query.order_by(NA_DB.create_date.desc())
    CHANNEL_ID = 'UCg7rkxrTnIhiHEpXY1ec9NA'

    # 채널 정보 가져오기
    channel_info = get_channel_info(CHANNEL_ID)
    # 값이 있음에 따라 HTML요소로 넘겨주기 
    if text and simir_text:
        return render_template("NA/result.html", result=result, table_list=table_list, result2=result2, cache_input=text, cache_output=result, channel_info=channel_info)
    elif text:
        return render_template("NA/result.html", result=result, table_list=table_list, cache_input=text, cache_output=result, channel_info=channel_info)
    elif simir_text:
        return render_template("NA/result.html", table_list=table_list, result2=result2, cache_input=text, cache_output=result, channel_info=channel_info)
    else :
        return render_template("NA/result.html", table_list=table_list, channel_info=channel_info)

@bp.route('/detail/<int:comment_id>', methods=['GET','POST'])
def detail(comment_id):
    comment = NA_DB.query.get(comment_id)
    return render_template("NA/comment_processing.html", comment=comment)


@bp.route('/delete/<int:comment_id>', methods=['GET','POST'])
def delete_comment(comment_id):
    comment = NA_DB.query.get(comment_id)
    db.session.delete(comment)
    db.session.commit()
    table_list = NA_DB.query.order_by(NA_DB.create_date.desc())
    return redirect(url_for('NA.show_result'))


@bp.route('/update/<int:comment_id>', methods=['GET','POST']) # 수정 추가기능 구현 필요
def update_comment(comment_id):
    comment = NA_DB.query.get(comment_id)
    comment.output = "아 ㅋㅋ수정된 댓글이라고 ㅋㅋㅋ"
    db.session.commit()
    return redirect(url_for('NA.show_result'))


@bp.route('/upload', methods=['GET','POST'])
def upload_comment():
    req_dict = request.form.to_dict() # 값들 갖고 오기 
    text = req_dict.get('cache_input')
    result= req_dict.get('cache_output')
    q = NA_DB(input=text, output=result, create_date=datetime.now())
    db.session.add(q)
    db.session.commit()
    return redirect(url_for('NA.show_result'))

