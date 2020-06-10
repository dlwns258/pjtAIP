# _*_coding: utf-8 _*_

from flask import Flask,request,jsonify,Response, make_response
from flask_restful import Resource,Api,reqparse
import numpy as np
import tensorflow.compat.v1 as tf
import model.ChakeList as md
import json
import pymysql
from flask_cors import CORS

tf.disable_v2_behavior()

app = Flask(__name__)
api = Api(app)

CORS(app)

db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='root', database='testdb', charset='utf8')

cursor = db.cursor()

#cursor.execute("CREATE TABLE opinion(name VARCHAR(255), colonoscopy int(10), gastroscopy int(10), tcd int(10), thorax int(10), thyroid int(10))")

class main(Resource):
    def get(self):
        try:

            return "Hello World"
        except Exception as e:
            app.logger.error(e)
            return {'error':str(e)}

class Index(Resource):
    def get(self):
        try:
            ls = [{'id': 1, 'name': 'kim'}, {'id': 2, 'name': 'lee'}]
            lsj = json.dumps(ls)
            resp = make_response(lsj)
            resp.mimetype = 'application/x-www-form-urlencoded'
            return jsonify(ls)
        except Exception as e:
            app.logger.error(e)
            return {'error':str(e)}
    def post(self):
        try:
            data = request.get_json()

            print(data)

            result1 = md.Colonoscopy_load(data)
            result2 = md.Gastroscopy_load(data)
            result3 = md.Tcd_load(data)
            result4 = md.Thorax_load(data)
            result5 = md.Thyroid_load(data)

            res = {"Colonoscopy": result1, "Gastroscopy": result2, "Tcd":result3,"Thorax":result4,"Thyroid":result5}

            sql ='INSERT INTO opinion (name, colonoscopy, gastroscopy, tcd, thorax, thyroid) VALUES(%s,%s,%s,%s,%s,%s)'

            cursor.execute(sql,(data['name'],result1,result2,result3,result4,result5))

            db.commit()

            print(res)

            return jsonify(res)

        except Exception as e:
            app.logger.error(e)
            return {'error':str(e)}

api.add_resource(main,'/')
api.add_resource(Index,'/TEST2')


if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5000")