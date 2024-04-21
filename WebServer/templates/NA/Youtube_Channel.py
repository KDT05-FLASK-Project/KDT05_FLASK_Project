# stack overflow : https://stackoverflow.com/questions/14366648/how-can-i-get-a-channel-id-from-youtube#:~:text=To%20obtain%20the%20channel%20id%20you%20can%20view,be%20the%20channel%20ID%20you%20are%20looking%20for.
from googleapiclient.discovery import build

# 유튜브 API 키
API_KEY = 'AIzaSyB9AhWjWm1e-BRA3eBCxsPxNG5evUQjBOk'

# API 클라이언트 빌드
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_channel_info(channel_id):
    # 채널 정보 가져오기
    request = youtube.channels().list(
        part='snippet,statistics',
        id=channel_id
    )
    response = request.execute()

    # 채널 정보 파싱
    channel = response['items'][0]
    channel_snippet = channel['snippet']
    channel_statistics = channel['statistics']

    # 필요한 정보 추출
    channel_title = channel_snippet['title']
    subscriber_count = channel_statistics['subscriberCount']
    video_count = channel_statistics['videoCount']
    channel_description = channel_snippet['description']

    return {
        'channel_title': channel_title,
        'subscriber_count': subscriber_count,
        'video_count': video_count,
        'channel_description': channel_description
    }

if __name__ == '__main__':
    # 유튜브 채널 ID
    CHANNEL_ID = 'UCg7rkxrTnIhiHEpXY1ec9NA'

    # 채널 정보 가져오기
    channel_info = get_channel_info(CHANNEL_ID)

    # 채널 정보 출력
    print('채널 제목:', channel_info['channel_title'])
    print('구독자 수:', channel_info['subscriber_count'])
    print('영상 개수:', channel_info['video_count'])
    print('채널 설명:', channel_info['channel_description'])

