import numpy as np
import pickle
import plotly
import json
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.cluster import KMeans
from flask import Flask, request, render_template,Markup
from keras.models import load_model


model=load_model(r"D:\data\Project_code-flask\models\crime_predict.h5")
app = Flask(__name__) #creating the Flask class object   
 
@app.route('/home.html') 
@app.route('/') 
def home():  
    return render_template('home.html')   
@app.route('/prediction.html') #decorator drfines the   
def prediction():  
    return render_template('prediction.html')    

@app.route('/predict',methods=["GET","POST"])
def res():
    if request.method=="POST":
        year=request.form.get('year').upper()
        state=request.form.get('state').upper()
        districts=request.form.get('districts').upper()
        a,b,c=state,districts,year
        year=np.array(year,ndmin=2)
        state=np.array(state,ndmin=1)
        districts=np.array(districts,ndmin=1)
        crime=request.form.get('crime')
        # Load the LabelEncoder object from the file
        with open('D:\data\Project_code-flask\models\state_encoder.pkl', 'rb') as file:
            state_encoder = pickle.load(file)
        
        with open('D:\data\Project_code-flask\models\district_encoder.pkl', 'rb') as file:
            district_encoder = pickle.load(file)
        state=state_encoder.transform(state)
        districts=district_encoder.transform(districts)
        in_scaler = MinMaxScaler(feature_range=(0, 1))
        in_scaler.min_, in_scaler.scale_ = np.load('D:\data\Project_code-flask\models\input_scaler.npy')
        year=in_scaler.transform(year)
        #Prediction
        state=np.array(state).reshape(-1,1)
        districts=np.array(districts).reshape(-1,1)
        x_test=np.concatenate((state,districts,year), axis=1)
        x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))
        x_test = np.asarray(x_test).astype('float32')
        result=model.predict(x_test)
        result=result.reshape(-1,7)
        # Data normalization using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.min_, scaler.scale_ = np.load('D:\data\Project_code-flask\models\scaler_params.npy')
        result=scaler.inverse_transform(result)
        cluster_names=['High','Moderate','Low']
        if (crime=="MURDER"):
            with open('D:\data\Project_code-flask\models\kmeans\kmeans_murder.pkl', 'rb') as file:
                kmeans = pickle.load(file)
            res=np.array(result[0][0], dtype=np.float32) 
            cluster=cluster_names[int(kmeans.predict(res.reshape(1,-1)))]
            return render_template('prediction.html',rate=cluster,a=a,b=b,c=c,d=crime,prediction=int(result[0][0]))
        elif(crime=="RAPE"):
            with open('D:\data\Project_code-flask\models\kmeans\kmeans_rape.pkl', 'rb') as file:
                kmeans = pickle.load(file)
            res=np.array(result[0][1], dtype=np.float32)
            cluster=cluster_names[int(kmeans.predict(res.reshape(1,-1)))]
            return render_template('prediction.html',rate=cluster,a=a,b=b,c=c,d=crime,prediction=int(result[0][1]))
        elif(crime=="KIDNAPPING"):
            with open('D:\data\Project_code-flask\models\kmeans\kmeans_kidnap.pkl', 'rb') as file:
                kmeans = pickle.load(file)
            res=np.array(result[0][2], dtype=np.float32)
            cluster=cluster_names[int(kmeans.predict(res.reshape(1,-1)))]
            return render_template('prediction.html',rate=cluster,a=a,b=b,c=c,d=crime,prediction=int(result[0][2]))
        elif(crime=="THEFT"):
            with open('D:\data\Project_code-flask\models\kmeans\kmeans_theft.pkl', 'rb') as file:
                kmeans = pickle.load(file)
            res=np.array(result[0][3], dtype=np.float32) 
            cluster=cluster_names[int(kmeans.predict(res.reshape(1,-1)))]
            return render_template('prediction.html',rate=cluster,a=a,b=b,c=c,d=crime,prediction=int(result[0][3]))
        elif(crime=="DOWRY_DEATHS"):
            with open('D:\data\Project_code-flask\models\kmeans\kmeans_dowry.pkl', 'rb') as file:
                kmeans = pickle.load(file)
            res=np.array(result[0][4], dtype=np.float32) 
            cluster=cluster_names[int(kmeans.predict(res.reshape(1,-1)))]
            return render_template('prediction.html',rate=cluster,a=a,b=b,c=c,d=crime,prediction=int(result[0][4]))
        elif(crime=="OTHER_CRIMES"):
            with open('D:\data\Project_code-flask\models\kmeans\kmeans_other.pkl', 'rb') as file:
                kmeans = pickle.load(file)
            res=np.array(result[0][5], dtype=np.float32) 
            cluster=cluster_names[int(kmeans.predict(res.reshape(1,-1)))]
            return render_template('prediction.html',rate=cluster,a=a,b=b,c=c,d=crime,prediction=int(result[0][5]))
        else:
            with open('D:\data\Project_code-flask\models\kmeans\kmeans_total.pkl', 'rb') as file:
                kmeans = pickle.load(file)
            res=np.array(result[0][6], dtype=np.float32) 
            cluster=cluster_names[int(kmeans.predict(res.reshape(1,-1)))]
            return render_template('prediction.html',rate=cluster,a=a,b=b,c=c,d=crime,prediction=int(result[0][6]))
    return render_template('prediction.html')
@app.route('/analysis.html') 
def analysis():  
    return render_template('analysis.html')  


@app.route('/analyse',methods=["GET","POST"])
def analyse():
    if request.method=="POST":
        year=request.form.get('year').upper()
        state=request.form.get('state').upper()
        districts=request.form.get('districts').upper()
        a,b,c=state,districts,year
        year=np.array(year,ndmin=2)
        state=np.array(state,ndmin=1)
        districts=np.array(districts,ndmin=1)
        crime=request.form.get('crime')
        # Load the LabelEncoder object from the file
        with open('D:\data\Project_code-flask\models\state_encoder.pkl', 'rb') as file:
            state_encoder = pickle.load(file)
        
        with open('D:\data\Project_code-flask\models\district_encoder.pkl', 'rb') as file:
            district_encoder = pickle.load(file)
        state=state_encoder.transform(state)
        districts=district_encoder.transform(districts)
        in_scaler = MinMaxScaler(feature_range=(0, 1))
        in_scaler.min_, in_scaler.scale_ = np.load('D:\data\Project_code-flask\models\input_scaler.npy')
        year=in_scaler.transform(year)
        #Prediction
        state=np.array(state).reshape(-1,1)
        districts=np.array(districts).reshape(-1,1)
        x_test=np.concatenate((state,districts,year), axis=1)
        x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))
        x_test = np.asarray(x_test).astype('float32')
        result=model.predict(x_test)
        result=result.reshape(-1,7)
        # Data normalization using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.min_, scaler.scale_ = np.load('D:\data\Project_code-flask\models\scaler_params.npy')
        result=scaler.inverse_transform(result)
        df1=pd.DataFrame({
            'crime':['Murder','Rape','Kidnapping','Theft','Dowry Deaths','Other crimes'],
            'rate':[int(result[0][0]),int(result[0][1]),int(result[0][2]),int(result[0][3]),int(result[0][4]),int(result[0][5])]
        })
        fig1=px.pie(df1,values='rate',names='crime')
        fig1.update_layout(title="Pie chart for all crimes")
        graph1=json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)
        df2=pd.DataFrame({
            'crime':['Murder','Rape','Kidnapping','Theft','Dowry Deaths'],
            'rate':[int(result[0][0]),int(result[0][1]),int(result[0][2]),int(result[0][3]),int(result[0][4])]
        })
        fig2=px.pie(df2,values='rate',names='crime')
        fig2.update_layout(title="Pie chart for major crimes")
        graph2=json.dumps(fig2,cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('analysis.html',graph1=graph1,graph2=graph2,a=a,b=b,c=c)
    return render_template('analysis.html')
@app.route('/city_analysis.html')   
def city_analysis():  
    return render_template('city_analysis.html') 
    
@app.route('/city_analyse',methods=["GET","POST"])
def city_analyse():
    if request.method=="POST":
        state1=request.form.get('state').upper()
        district=request.form.get('districts').upper()
        a,b=state1,district
        list1=[2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028]
        year=np.array(list1);
        state=np.empty(len(year),dtype='O')
        districts=np.empty(len(year),dtype='O')
        for i in range(0,len(year)):
            state[i]=state1
            districts[i]=district
        year=np.array(year).reshape(-1,1)
        state=np.array(state,ndmin=1)
        districts=np.array(districts,ndmin=1)
        # Load the LabelEncoder object from the file
        with open('D:\data\Project_code-flask\models\state_encoder.pkl', 'rb') as file:
            state_encoder = pickle.load(file)
        
        with open('D:\data\Project_code-flask\models\district_encoder.pkl', 'rb') as file:
            district_encoder = pickle.load(file)
        state=state_encoder.transform(state)
        districts=district_encoder.transform(districts)
        in_scaler = MinMaxScaler(feature_range=(0, 1))
        in_scaler.min_, in_scaler.scale_ = np.load('D:\data\Project_code-flask\models\input_scaler.npy')
        year=in_scaler.transform(year)
        #Prediction
        state=np.array(state).reshape(-1,1)
        districts=np.array(districts).reshape(-1,1)
        x_test=np.concatenate((state,districts,year), axis=1)
        x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))
        x_test = np.asarray(x_test).astype('float32')
        result=model.predict(x_test)
        result=result.reshape(-1,7)
        # Data normalization using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.min_, scaler.scale_ = np.load('D:\data\Project_code-flask\models\scaler_params.npy')
        result=scaler.inverse_transform(result)
        df1=pd.DataFrame({
            'year':['2018','2019','2020','2021','2022','2023','2024','2025','2026','2027','2028'],
            'rate':[int(result[0][0]),int(result[1][0]),int(result[2][0]),int(result[3][0]),int(result[4][0]),int(result[5][0]),int(result[6][0]),int(result[7][0]),int(result[8][0]),int(result[9][0]),int(result[10][0])]
        })
        fig1=px.bar(df1,x='year',y='rate')
        fig1.update_layout(title="Murder Crime pattern over the years")
        graph1=json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)
        
        df2=pd.DataFrame({
            'year':['2018','2019','2020','2021','2022','2023','2024','2025','2026','2027','2028'],
            'rate':[int(result[0][1]),int(result[1][1]),int(result[2][1]),int(result[3][1]),int(result[4][1]),int(result[5][1]),int(result[6][1]),int(result[7][1]),int(result[8][1]),int(result[9][1]),int(result[10][1])]
        })
        fig2=px.bar(df2,x='year',y='rate')
        fig2.update_layout(title="Rape Crime pattern over the years")
        graph2=json.dumps(fig2,cls=plotly.utils.PlotlyJSONEncoder)
        
        df5=pd.DataFrame({
            'year':['2018','2019','2020','2021','2022','2023','2024','2025','2026','2027','2028'],
            'rate':[int(result[0][4]),int(result[1][4]),int(result[2][4]),int(result[3][4]),int(result[4][4]),int(result[5][4]),int(result[6][4]),int(result[7][4]),int(result[8][4]),int(result[9][4]),int(result[10][4])]
        })
        fig5=px.bar(df5,x='year',y='rate')
        fig5.update_layout(title="Dowry Deaths Crime pattern over the years")
        graph5=json.dumps(fig5,cls=plotly.utils.PlotlyJSONEncoder)
        
        df6=pd.DataFrame({
            'year':['2018','2019','2020','2021','2022','2023','2024','2025','2026','2027','2028'],
            'rate':[int(result[0][6]),int(result[1][6]),int(result[2][6]),int(result[3][6]),int(result[4][6]),int(result[5][6]),int(result[6][6]),int(result[7][6]),int(result[8][6]),int(result[9][6]),int(result[10][6])]
        })
        fig6=px.bar(df6,x='year',y='rate')
        fig6.update_layout(title="Total Crime pattern over the years")
        graph6=json.dumps(fig6,cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('city_analysis.html',graph1=graph1,graph2=graph2,graph5=graph5,graph6=graph6,a=a,b=b)
    return render_template('city_analysis.html')
 
@app.route('/year_analysis.html') 
def year_analysis():  
    return render_template('year_analysis.html')

@app.route('/registration.html') 
def reg():  
    return render_template('registration.html')

@app.route('/password_change.html') 
def f_p():  
    return render_template('password_change.html')
@app.route('/user.html') 
def user():  
    return render_template('user.html')


if __name__ =='__main__':  
    app.run(debug = True)  
