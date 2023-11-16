# python
from flask import Flask, request, render_template, redirect, jsonify, session
from flask_cors import CORS
import requests
import json
import os
import re
from collections import Counter

# config
import config

# youtube API
from youtube_transcript_api import YouTubeTranscriptApi

# gpt API
import openai

# DB
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from models import db, User, fakeVideo

#wiki
from wiki import wiki_page

#naver
from naver_news import get_news

#langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain.chains.mapreduce import MapReduceChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
)
openai_api_key = config.GPT_KEY
llm = ChatOpenAI(temperature=0, model_name=config.GPT_MODEL, openai_api_key=config.GPT_KEY)

# Map prompt
map_template = """The following is a set of documents
{docs}
Based on this list of docs, Please summarize the contents in detail
Helpful Answer:"""

map_prompt = PromptTemplate.from_template(map_template)

# Map chain
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce prompt
reduce_template = """The following is set of summaries:
{doc_summaries}
Take these and distill it into a final, consolidated summary of the main summary.
The final answer should be written in detail and in Korean.
Helpful Answer:"""

reduce_prompt = PromptTemplate.from_template(reduce_template)

# Reduce chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="doc_summaries"
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)



# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)


##flask config
app = Flask(__name__)
CORS(app)

basedir = os.path.abspath(os.path.dirname(__file__))  # 현재있는 파일의 디렉토리 절대경로
dbfile = os.path.join(basedir, "db.sqlite")  # basdir 경로안에 DB파일 만들기

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + dbfile  # 내가 사용 할 DB URI
app.config["SQLALCHEMY_COMMIT_ON_TEARDOWN"] = True  # 비지니스 로직이 끝날때 Commit 실행(DB반영)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False  # 수정사항에 대한 TRACK
app.config["SECRET_KEY"] = "1234"  # SECRET_KEY

openai.api_key = config.GPT_KEY
# db session create
engine = create_engine("sqlite:///" + dbfile, echo=True)
Session = sessionmaker()
Session.configure(bind=engine)

db.init_app(app)
db.app = app
with app.app_context():
    db.create_all()



#main controller
@app.route('/')
def index():  # put application's code here
    return render_template("index.html")


@app.route('/create', methods=["get"])
def create_get():
    return render_template("create.html")


@app.route('/create', methods=["post"])
def create_post():
    input_id = request.form["username"]
    input_password = request.form["password"]

    db_session = Session()
    user = User(
        userid=input_id,
        password=input_password,
        refresh_token=None,
        access_token=None,
    )
    db_session.merge(user)
    db_session.commit()
    db_session.close()

    return redirect("/", code=302)


@app.route('/createCheck', methods=["get"])
def create_check():
    db_session = Session()
    temp = db_session.query(User).all()
    for i in temp:
        print(i.userid, i.password)

    return redirect("/", code=302)


@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    print(username)
    print(password)

    db_session = Session()
    user = db_session.query(User).filter(User.userid == username).first()
    error = "성공"
    if not user:
        error = "존재하지 않는 사용자입니다."
    elif user.password != password:
        error = "비밀번호가 올바르지 않습니다."
    print(error)
    return redirect(config.auth_url, code=302)


# react api server
@app.route("/link", methods=["POST"])
def link():
    try:
        data = request.json
        result = {'message': 'JSON 데이터를 성공적으로 처리했습니다.'}
        match = re.search(r'[?&]v=([^&]+)', data["data"])
        if match:
            result = {'message': match.group(1)}
        else:
            result = {'message': ""}
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route("/script", methods=["POST"])
def script():
    response = {"script": "스크립트가 존재하지 않습니다."}
    try:
        data = request.json
        print(data["data"])
        srts = YouTubeTranscriptApi.get_transcript(data["data"], languages=["ko"])
        srt = ""
        for i in srts:
            srt += i["text"]
        response = {"script": srt}
        return jsonify(response), 200
    except Exception as e:
        return jsonify(response), 200

@app.route("/summary", methods=["POST"])
def summary():
    response = {"summary": "", "keyword": []}
    try:
        data = request.json
        docs = [Document(page_content=x) for x in text_splitter.split_text(data["data"])]
        split_docs = text_splitter.split_documents(docs)
        sum_result = map_reduce_chain.run(split_docs)
        response["summary"] = sum_result
        # summary_messages = [
        #     {
        #         "role": "system",
        #         # "content": "해당 영상 자막을 바탕으로 내용을 요약해줘, 반드시 300자 내외로 한국어로 대답해줘",
        #         "content": 'Please summarize the contents based on the subtitles of the video, summarize them in detail and answer them in Korean',
        #     },
        #     {"role": "user", "content": data["data"]},
        # ]
        # summary_response = openai.ChatCompletion.create(model=config.GPT_MODEL_4, messages=summary_messages)
        # print(summary_response)
        # summarys = summary_response["choices"][0]["message"]["content"]
        # response["summary"] = summarys

        keyword_messages = [
            {
                "role": "system",
                # "content": "해당 요약본을 바탕으로 해당 영상에 대해 정보를 얻기 위해 검색할 키워드를 ,으로 구분해서 알려줘",
                "content": 'Based on the summary, tell us the keywords you want to search for to get information about the video, separated by ,.For example, please answer with Korean keywords in this format: “word, word, word.”',
            },
            {"role": "user", "content": sum_result},
        ]
        keyword_response = openai.ChatCompletion.create(model=config.GPT_MODEL, messages=keyword_messages)
        keyword = keyword_response["choices"][0]["message"]["content"].split(",")
        # response["keyword"] = ["영국", "한국"]
        response["keyword"] = keyword
        return jsonify(response), 200
    except Exception as e:
        print(e)
        return jsonify(response), 200


@app.route("/wiki", methods=["POST"])
def wiki():
    response = {"wiki": "사전 데이터가 없습니다."}
    wikis = []
    try:
        data = request.json
        for i in data["data"][:3]: #3개
            page = wiki_page(i)
            if page == None:
                continue
            wikis.append(page.summary)
        response["wiki"] = wikis
        return jsonify(response), 200
    except Exception as e:
        return jsonify(response), 200


@app.route("/naver", methods=["POST"])
def naver():
    response = {"naver": ["네이버 데이터가 없습니다."]}
    navers = []
    try:
        data = request.json
        for i in data["data"][:3]: #3개
            print(i)
            page = get_news(i)
            if page == None:
                continue
            navers += page
        response["naver"] = navers
        print(response)
        return jsonify(response), 200
    except Exception as e:
        return jsonify(response), 200

answer_schema = {
    "type": "object",
    "properties": {
        "reason" : {
            "type": "string",
            "description": "판단이유"
        },
        "reliability" : {
            "type": "string",
            "description": "신뢰도를 백분률로 %없이 숫자로 알려줘"
        },
        "Discrimination" : {
            "type": "string",
            "description": "가짜일 경우 1, 진짜일 경우 0"}
        },
    "required": ["reason", "reliability", "Discrimination"]
}
@app.route("/detected", methods=["POST"])
def detected():
    response = {"detected": ""}
    try:
        data = request.json
        print(data["summary"])
        news = data["naver"]
        wiki = data["wiki"]
        summary = data["summary"]
        summary_messages = [
            {
                "role": "system",
                "content": '''사용자가 제공하는 요약 내용이 가짜 뉴스인지 판단하여 판단이유,신뢰도,가짜여부를 json형식에 맞게 한글로 대답해줘
                ''',
            },
            {"role": "user", "content": f"관련하여 뉴스정보는 다음과 같아[{naver}] 해당 정보를 바탕으로 다음 요약정보 {summary}를 판단해줘" },
        ]
        detected_response = openai.ChatCompletion.create(model=config.GPT_MODEL_4, messages=summary_messages, functions=[{
            "name": "get_answer",
            "description": "사용자가 제공하는 요약 내용이 가짜 뉴스인지 판단하여 판단이유,신뢰도,가짜여부를 json형식에 맞게 한글로 대답해줘",
            "parameters": answer_schema}],
        function_call={
            "name": "get_answer"
            },
        )
        # detected = detected_response["choices"][0]["message"]["content"]
        detected = detected_response["choices"][0]["message"]["function_call"]["arguments"]
        print(detected)
        response["detected"] = detected
        return jsonify(response), 200
    except Exception as e:
        print(e)
        return jsonify(response), 200

@app.route("/savefakeVideo", methods=["POST"])
def savefakeVideo():
    response = {'message': 'JSON 데이터를 성공적으로 처리했습니다.'}
    try:
        data = request.json
        db_session = Session()
        info = {
            "description" : data["description"],
            "script":data["script"],
            "summary":data["summary"],
            "keyword":data["keyword"],
            "wiki":data["wiki"],
            "naver":data["naver"],
            "reason":data["reason"],
        }

        json_info = json.dumps(info, ensure_ascii=False)

        video = fakeVideo(
            id = data["id"],
            title= data["title"],
            channelTitle=data["channelTitle"],
            reliability=data["reliability"],
            data = json_info,
            discrimination = data["Discrimination"],
            viewCount =  data["viewCount"],
            likeCount =  data["likeCount"],
            channelId = data["channelId"],
        )
        db_session.merge(video)
        db_session.commit()
        db_session.close()

        return jsonify(response), 200
    except Exception as e:
        print(e)
        return jsonify(response), 200


@app.route("/getfakeVideo", methods=["POST"])
def getfakeVideo():
    data = request.json
    target_id = data["data"]
    print(target_id)
    db_session = Session()
    news_data = db_session.query(fakeVideo).filter_by(id = target_id).first()
    json_info = {"id": None}
    if news_data:
        json_info = json.loads(news_data.data)
        json_info["discrimination"] = news_data.discrimination
        json_info["viewCount"] = news_data.viewCount
        json_info["likeCount"] = news_data.likeCount
        json_info["title"] = news_data.title
        json_info["channelTitle"] = news_data.channelTitle
        json_info["reliability"] = news_data.reliability
        json_info["channelId"] = news_data.channelId
        return json_info
    return json_info


@app.route("/getVideoList", methods=["POST"])
def getVideoList():

    db_session = Session()
    news_data = db_session.query(fakeVideo).all()
    infos = []
    for video in news_data:
        video_data = {
            "id": video.id,
            "data": json.loads(video.data),  # JSON 문자열을 파이썬 객체로 변환
            "discrimination": video.discrimination,
            "viewCount": video.viewCount,
            "likeCount": video.likeCount,
            "title": video.title,
            "channelTitle": video.channelTitle,
            "reliability": video.reliability
        }
        infos.append(video_data)
    json_info = {"data": infos}
    return json_info


@app.route("/getFakeList", methods=["POST"])
def getFakeList():

    db_session = Session()
    news_data = db_session.query(fakeVideo).filter(fakeVideo.discrimination == 1).order_by(fakeVideo.viewCount.desc()).all()
    infos = []
    for video in news_data:
        video_data = {
            "id": video.id,
            "data": json.loads(video.data),  # JSON 문자열을 파이썬 객체로 변환
            "discrimination": video.discrimination,
            "viewCount": video.viewCount,
            "likeCount": video.likeCount,
            "title": video.title,
            "channelTitle": video.channelTitle,
            "reliability": video.reliability
        }
        infos.append(video_data)
    json_info = {"data": infos}
    return json_info

@app.route("/getFakeChannelList", methods=["POST"])
def getFakeChannelList():

    db_session = Session()
    result = db.session.query(fakeVideo.channelTitle, fakeVideo.channelId, db.func.count(fakeVideo.id)).filter(
        fakeVideo.discrimination == 1).group_by(fakeVideo.channelTitle).order_by(db.func.count(fakeVideo.id).desc()).all()
    for row in result:
        print(f"Channel Title: {row[0]}, Count: {row[1]}, {row[2]}")
    infos = []
    for row in result:
        video_data = {
            "channelTitle": row[0],
            "channelId": row[1],
            "count": row[2],
        }
        infos.append(video_data)
    json_info = {"data": infos}
    return json_info

@app.route("/getFakeVideoByChannel", methods=["POST"])
def getFakeVideoByChannel():

    db_session = Session()
    data = request.json
    target_id = data["data"]
    news_data = db_session.query(fakeVideo).filter_by(channelId=target_id).order_by(fakeVideo.viewCount.desc()).all()
    discrimination_counts = Counter(record.discrimination for record in news_data)

    infos = []
    for video in news_data:
        video_data = {
            "id": video.id,
            "data": json.loads(video.data),  # JSON 문자열을 파이썬 객체로 변환
            "discrimination": video.discrimination,
            "viewCount": video.viewCount,
            "likeCount": video.likeCount,
            "title": video.title,
            "channelTitle": video.channelTitle,
            "reliability": video.reliability,
        }
        infos.append(video_data)
    json_info = {"data": infos, "discrimination_counts" : discrimination_counts[1],
            "counts" : len(news_data)}
    return json_info

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=config.port)
