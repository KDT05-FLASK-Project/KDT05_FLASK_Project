from flask import Blueprint, render_template, request 
from datetime import datetime
import torch
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration
from transformers import pipeline
import re
import contractions

def str_preprocessing(x: str):
    """ 자연어 처리 함수 """
    
    ### str이 아니면 그냥 반환
    if not isinstance(x, str): 
        return x
    
    ### 소문자로 변환
    x = x.lower()

    ### 소괄호로 둘러싸인 문자열 삭제
    pattern1 = r'\([^)]*\)'
    x = re.sub(pattern1, '', x)

    ### 대괄호로 둘러싸인 문자열 삭제
    pattern2 = r'\[[^\]]*\]'
    x = re.sub(pattern2, '', x)

    ### 축약어, 슬랭 처리
    x = contractions.fix(x)

    return x


def get_summary(input_sentence: str, preprocessing=True):
    """ 요약 문자열 생성 함수 """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "first" if not preprocessing else "second"
    
    ### 저장된 모델 불러오기(모델 객체 생성 -> 상태 사전 불러오기)
    model = BartForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path="facebook/bart-base"
    ).to(device)
    state_dict = torch.load(f'변주영/models/{model_type}/Latest_Bart_Amazon_Books.pt',
                                map_location=torch.device(device))  # collections.OrderedDict
    model.load_state_dict(state_dict)

    ### 평가 모드 ON
    model.eval()

    ### 토크나이저 생성
    tokenizer = BartTokenizer.from_pretrained(
        pretrained_model_name_or_path="facebook/bart-base"
    )

    ### 파이프라인 생성
    summarizer = pipeline(
        task="summarization",
        model=model,    # 모델 설정
        tokenizer=tokenizer,
        max_length=50,  # 입력 텍스트가 짧을 경우, 더 줄여도 된다. (예: 24)
        device="cpu"
    )

    ### predict 수행
    if preprocessing:   # second 모델일 경우 사용
        input_sentence = str_preprocessing(input_sentence)
    summarizer_result = summarizer(input_sentence)
    predicted_summarization = summarizer_result[0]["summary_text"]
    return predicted_summarization


bp = Blueprint(name='JY', import_name=__name__, template_folder = 'templates',
                    url_prefix="/jy_db")

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_sentence = request.form['sentence']
        summary_result = get_summary(input_sentence)
        return render_template("JY/index.html", input_sentence=input_sentence, summary_result=summary_result)
    
    ### GET 요청일 경우 기본 페이지를 렌더링
    return render_template("JY/index.html") # templates 함수 내에서 파일을 찾음
