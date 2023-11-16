# 네이버 검색 API 예제 - 블로그 검색
import os
import sys
import urllib.request
import json
import html
import re

import config

client_id = config.naver_client_id
client_secret = config.naver_client_secret


def get_news(content):
    encText = urllib.parse.quote(content)
    print(encText)
    url = "https://openapi.naver.com/v1/search/news?display=10&start=1&sort=sim&query=" + encText  # JSON 결과
    # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if rescode == 200:
        response_body = response.read()
        body = json.loads(response_body.decode("utf-8"))
        print(body)
        answer = []
        for i in body["items"][:2]:
            answer.append(re.sub(r"<\/?b>", "", html.unescape(i["description"])))
        return answer
    else:
        return None
