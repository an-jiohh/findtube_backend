import wikipediaapi


def wiki_page(search):  # 검색어는 계속 바뀌므로 함수로 정의
    wiki = wikipediaapi.Wikipedia("MyProjectName","ko")
    page = wiki.page(search)  # 검색어에 해당되는 페이지 전체 가져오기
    if page.exists() == False:
        return None
    print("제목: ", page.title)  # 제목확인, 요약, url 확인
    print("요약: ", page.summary)
    print("url: ", page.fullurl)
    print(page.text)  # 전체 페이지 내용 보여주기
    return page

