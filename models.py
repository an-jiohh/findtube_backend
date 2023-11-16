from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tableName__ = "video"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    userid = db.Column(db.String(32))
    password = db.Column(db.String(32))
    refresh_token = db.Column(db.String(32))
    access_token = db.Column(db.String(32))

class fakeVideo(db.Model):
    __tablename__ = 'fakevideo'

    id = db.Column(db.String, primary_key=True)
    data = db.Column(db.String)
    discrimination = db.Column(db.Integer)
    viewCount = db.Column(db.Integer)
    likeCount = db.Column(db.Integer)
    title = db.Column(db.String)
    channelTitle = db.Column(db.String)
    reliability = db.Column(db.String)
    channelId = db.Column(db.String)