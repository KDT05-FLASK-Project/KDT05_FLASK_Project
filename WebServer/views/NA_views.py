from flask import Blueprint, render_template, request 
from datetime import datetime
from WebServer.models import NA_DB
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch 
from gensim.models import Word2Vec

# 댓글 생성 함수 선언
def generate_text(text):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    model = torch.load("명노아/model/model3.pth", map_location=torch.device('cpu'))
    model.eval()
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
    
    if text : # 댓글 생성 함수
        result = generate_text(text)
  
    if simir_text: # 유사도 함수
        word2vec = Word2Vec.load("명노아/model/Gensim.model")
        result2 = word2vec.wv.most_similar(simir_text, topn=5)

    # 테이블 값 갖고오기 
    table_list = NA_DB.query.order_by(NA_DB.create_date.desc())
    
    
    
    
    
    
    
    # 값이 있음에 따라 HTML요소로 넘겨주기 
    if text and simir_text:
        return render_template("NA/result.html", result=result, table_list=table_list, result2=result2)
    elif text:
        return render_template("NA/result.html", result=result, table_list=table_list)
    elif simir_text:
        return render_template("NA/result.html", table_list=table_list, result2=result2)






