# KDT05-FLASK Project

<hr/>

<hr/>

경북대학교 KDT(Korea Digital Training) 빅데이터 전문가 양성과정 5기 : Flask 3팀입니다

변주영 : [깃허브 링크](https://github.com/5amriley)  
이시명 : [깃허브 링크](https://github.com/juugii-ho)  
양현우 : [깃허브 링크](https://github.com/daat1996)  
명노아 : [깃허브 링크](https://github.com/noah2397)

![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![MySQL](https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)

<hr/>

#### 개발환경

| 패키지 이름      | 버전   |
| ---------------- | ------ |
| torchaudio       | 2.2.2  |
| torchtext        | 0.17.2 |
| torchvision      | 0.17.2 |
| tabulate         | 0.9.0  |
| python           | 3.8.19 |
| flask            | 3.0.3  |
| flask-migrate    | 4.0.7  |
| flask-sqlalchemy | 3.1.1  |

<hr/>

### 실행 방법

해당 소스 코드가 있는 폴더로 이동 후, 다음과 같은 커맨드를 콘솔 창에 입력

```
flask db init
flask db migrate
flask db upgrade
flask run
```

### 실행 방법(Docker)

```
docker-compose up -d
docker exec -it kdt05_flask_app_1 /bin/bash
cd /app/
flask db init
flask db migrate
flask db upgrade
cd /app/app/
python app.py
```

### KDT(Korea Digital Training)-HTML 활용(FLASK)

<hr/>

#### 사용한 데이터 사이트

1. AI-HUB : [링크](https://aihub.or.kr/)
2. Amazon Books : [링크](https://www.amazon.com/books-used-books-textbooks/b?node=283155)
3. 네이버 영화 리뷰 말뭉치 : [링크](https://github.com/e9t/nsmc)
4. 잠뜰 TV(유튜브 채널) : [링크](https://www.youtube.com/user/sleepground)
<hr/>

###### [주제입력]

- 목차

* 1. 주제 선정 배경
* 2. 역할 분담
* 3. 이시명(말뭉치 기반 다국어 번역 프로그램)
* 4. 변주영(아마존 독서 리뷰 요약)
* 5. 양현우(호불호 분석 프로그램)
* 6. 명노아(잠뜰TV 댓글 생성 프로그램)
* 7. 결론 & 시연
  </hr>

###### 역할 분담

|         역할         | 이시명 | 명노아 | 변주영 | 양현우 |
| :------------------: | :----: | :----: | :----: | :----: |
| 데이터 수집 & 전처리 |   ✅   |   ✅   |   ✅   |   ✅   |
|      모델 생성       |   ✅   |   ✅   |   ✅   |   ✅   |
|  전체 웹페이지 구축  |        |   ✅   |        |   ✅   |
|   개인 페이지 구축   |   ✅   |   ✅   |   ✅   |   ✅   |
|        Readme        |   ✅   |        |   ✅   |        |
