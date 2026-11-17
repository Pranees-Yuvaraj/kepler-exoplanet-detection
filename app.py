import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

st.set_page_config(page_title='Kepler | Exoplanet Intelligence', page_icon='🪐', layout='wide', initial_sidebar_state='expanded')
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:'Inter',sans-serif;}
.stApp {background:#f6f8fb;color:#172337;}
section[data-testid="stSidebar"] {background:#ffffff;border-right:1px solid #e4e9f1;}
section[data-testid="stSidebar"] > div {padding-top:1.3rem;}
.block-container {padding-top:2rem;max-width:1260px;padding-bottom:4rem;}
h1,h2,h3 {letter-spacing:-.045em;color:#172337;}
h1 {font-weight:800;}
p, label, div[data-testid="stCaptionContainer"] {color:#4e5e73;}
div[data-testid="stMetric"] {background:#ffffff;border:1px solid #e3e9f2;padding:20px;border-radius:15px;box-shadow:0 4px 16px #182b4608;}
div[data-testid="stMetricLabel"] {color:#66778c;}
div[data-testid="stMetricValue"] {color:#172337;}
section[data-testid="stSidebar"] div.stButton {margin:0 0 5px 0;}
section[data-testid="stSidebar"] div.stButton > button {width:100%;justify-content:flex-start;text-align:left;min-height:45px;padding:0 14px;border-radius:11px;border:1px solid transparent;background:transparent;color:#43546b;font-size:14px;font-weight:550;box-shadow:none;transition:background .15s ease;}
section[data-testid="stSidebar"] div.stButton > button:hover {background:#f0f5f9;border-color:#e2eaf1;color:#117e82;}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {background:#e3f4f2;border:1px solid #c5e8e4;color:#086d70;font-weight:750;box-shadow:inset 3px 0 0 #15999a;}
.stButton > button[kind="primary"] {background:#0d8587;color:white;border:1px solid #0d8587;border-radius:10px;}
.stButton > button[kind="primary"]:hover {background:#096c6f;color:white;}
.stButton > button, .stDownloadButton > button {border-radius:10px;}
[data-testid="stFileUploader"] {background:white;border:1px solid #e2e9f1;border-radius:14px;padding:12px;}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {border-radius:14px;overflow:hidden;}
[data-testid="stSidebar"] hr {border-color:#edf0f5;}
.kepler-brand {padding:6px 6px 16px;}
.kepler-brand strong {font-size:22px;font-weight:800;color:#152e3a;letter-spacing:-.06em;}
.kepler-brand small {display:block;color:#82909d;letter-spacing:.12em;font-size:10px;font-weight:700;margin-top:3px;}
.sidebar-label {color:#94a0b0;font-size:10px;font-weight:800;letter-spacing:.12em;padding:9px 10px 10px;}
.status-line {font-size:12px;color:#607184;margin:7px 10px;}
.status-dot {display:inline-block;width:8px;height:8px;border-radius:50%;background:#aab5c2;margin-right:9px;}
.status-dot.ok {background:#18a48c;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def parse_csv(content):
    df=pd.read_csv(io.BytesIO(content), low_memory=False)
    df.columns=[str(c).strip() for c in df.columns]
    if df.empty: raise ValueError('The CSV contains no rows.')
    if len(set(df.columns)) != len(df.columns): raise ValueError('Duplicate column names detected.')
    return df

def label_column(df):
    return next((c for c in df.columns if c.upper() in ('LABEL','CLASS','TARGET')),None)

def class_name(value):
    try:
        numeric=float(value)
        if numeric==2: return 'Exoplanet'
        if numeric==1: return 'Non-exoplanet'
    except (TypeError,ValueError):
        pass
    return 'Unknown'

def prepare(df, label_required=True):
    label=label_column(df)
    if label_required and label is None: raise ValueError('A LABEL column is required for training and evaluation.')
    features=df.drop(columns=[label]) if label else df.copy()
    features=features.apply(pd.to_numeric,errors='coerce')
    features=features.loc[:,features.notna().any(axis=0)]
    if features.shape[1]<2: raise ValueError('At least two numeric flux columns are required.')
    if features.shape[1]>5000: raise ValueError('More than 5000 flux columns is not supported in this build.')
    y=None
    if label:
        raw=pd.to_numeric(df[label],errors='coerce')
        if raw.isna().any() or not raw.isin([1,2]).all(): raise ValueError('Kepler labels must be 1 (non-host) or 2 (host).')
        y=(raw==2).astype(int)
    return features,y

def extract(x):
    a=x.to_numpy(dtype=float)
    with np.errstate(all='ignore'):
        return pd.DataFrame({'mean':np.nanmean(a,axis=1),'std':np.nanstd(a,axis=1),
          'min':np.nanmin(a,axis=1),'max':np.nanmax(a,axis=1),
          'p05':np.nanpercentile(a,5,axis=1),'p25':np.nanpercentile(a,25,axis=1),
          'median':np.nanmedian(a,axis=1),'p75':np.nanpercentile(a,75,axis=1),
          'p95':np.nanpercentile(a,95,axis=1),
          'range':np.nanmax(a,axis=1)-np.nanmin(a,axis=1)})

def plot_flux(row,title):
    vals=pd.to_numeric(row,errors='coerce').to_numpy(dtype=float)
    fig=go.Figure(go.Scatter(x=np.arange(len(vals)),y=vals,mode='lines',line=dict(color='#0d9999',width=1.15)))
    fig.update_layout(title=title,template='plotly_white',paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',height=365,
       margin=dict(l=25,r=20,t=55,b=35),xaxis_title='Measurement index',yaxis_title='Flux',font=dict(color='#43546b'))
    return fig

if 'train' not in st.session_state: st.session_state.train=None
if 'test' not in st.session_state: st.session_state.test=None
if 'model' not in st.session_state: st.session_state.model=None
if 'columns' not in st.session_state: st.session_state.columns=None

if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

with st.sidebar:
    st.markdown('<div class="kepler-brand"><strong>◈ KEPLER</strong><small>EXOPLANET INTELLIGENCE</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
    navigation = [
        ('Overview','▦  Overview'),
        ('Data Library','▤  Data Library'),
        ('Star Explorer','✦  Star Explorer'),
        ('Light Curves','⌁  Light Curves'),
        ('Model Training','◇  Model Training'),
        ('Predictions','◎  Predictions'),
        ('About','ⓘ  About'),
    ]
    for target, label in navigation:
        if st.button(label, key='nav_'+target.replace(' ','_'), type='primary' if st.session_state.page == target else 'secondary', use_container_width=True):
            st.session_state.page = target
            st.rerun()
    st.divider()
    st.markdown('<div class="sidebar-label">WORKSPACE STATUS</div>', unsafe_allow_html=True)
    for label, ready in [('Training dataset',st.session_state.train is not None),('Testing dataset',st.session_state.test is not None),('ML model',st.session_state.model is not None)]:
        st.markdown(f'<div class="status-line"><span class="status-dot {"ok" if ready else ""}"></span>{label}</div>', unsafe_allow_html=True)
    st.divider()
    st.caption('Kepler light curve analysis · v2.0')

page = st.session_state.page

st.title(page)
st.caption('Exoplanet Transit Detection Using Machine Learning')

if page=='Overview':
    st.markdown('### Discover patterns in starlight')
    st.write('Import Kepler-derived observations, explore stellar light curves, train a supervised classifier, and inspect model predictions in one workspace.')
    a,b,c=st.columns(3)
    a.metric('Training observations',len(st.session_state.train) if st.session_state.train is not None else '—')
    b.metric('Test observations',len(st.session_state.test) if st.session_state.test is not None else '—')
    c.metric('Model status','Ready' if st.session_state.model is not None else 'Not trained')
    st.info('Start in Data Library: upload exoTrain.csv and exoTest.csv. No synthetic dataset is needed.')
    st.markdown('#### Workflow')
    st.write('**01** Upload datasets  →  **02** Inspect light curves  →  **03** Train model  →  **04** Review predictions')

elif page=='Data Library':
    st.markdown('### Upload Kepler CSV datasets')
    st.write('Supported format: `LABEL` (1 = non-exoplanet class, 2 = exoplanet-host class) plus numeric flux columns. The uploader also accepts a custom numeric flux CSV for visualization.')
    for name,desc in [('train','Training CSV (exoTrain.csv)'),('test','Testing CSV (exoTest.csv)')]:
        uploaded=st.file_uploader(desc,type=['csv'],key='upload_'+name)
        if uploaded is not None:
            try:
                data=parse_csv(uploaded.getvalue())
                x,y=prepare(data)
                st.session_state[name]=data
                st.success(f'{name.title()} data loaded: {len(data):,} rows and {x.shape[1]:,} flux columns.')
            except Exception as e: st.error(f'Cannot load {name} CSV: {e}')
    for name in ('train','test'):
        data=st.session_state[name]
        if data is not None:
            with st.expander(f'{name.title()} dataset preview ({len(data):,} rows)',expanded=name=='train'):
                st.dataframe(data.iloc[:25,:min(12,len(data.columns))],use_container_width=True)
                x,y=prepare(data)
                st.write('Class counts:',y.value_counts().rename(index={0:'Non-host',1:'Exoplanet-host'}).to_dict())

elif page=='Star Explorer':
    st.markdown('Explore the uploaded observations, review class counts, and open a selected star’s light curve.')
    datasets={k:st.session_state[k] for k in ('train','test') if st.session_state[k] is not None}
    if not datasets:
        st.warning('Upload a dataset in Data Library first.')
    else:
        catalog_parts=[]
        for dataset_name,data in datasets.items():
            label=label_column(data)
            catalog=pd.DataFrame({
                'Dataset':dataset_name.title(),
                'Star #':np.arange(1,len(data)+1),
                'Classification':data[label].map(class_name) if label else 'Unlabeled',
            })
            id_col=next((c for c in data.columns if c.strip().lower() in ('kepid','kepler_id','kic','star_id','targetid')),None)
            if id_col:
                catalog.insert(2,'Star ID',data[id_col].astype(str).to_numpy())
            catalog_parts.append(catalog)

        stars=pd.concat(catalog_parts,ignore_index=True)
        total=len(stars)
        training_count=len(st.session_state.train) if st.session_state.train is not None else None
        exoplanet_count=int((stars['Classification']=='Exoplanet').sum())
        normal_count=int((stars['Classification']=='Non-exoplanet').sum())
        a,b,c,d=st.columns(4)
        a.metric('Total Observations',f'{total:,}')
        b.metric('Training Stars',f'{training_count:,}' if training_count is not None else '—')
        c.metric('Exoplanet Class',f'{exoplanet_count:,}')
        d.metric('Non-Exoplanet Class',f'{normal_count:,}')
        st.caption('Counts reflect rows in the uploaded CSV files. They are observations in this dataset, not unique astronomical stars verified by Kepler IDs.')

        left,right=st.columns([1,1.6])
        with left:
            st.subheader('Star Classification Distribution')
            st.bar_chart(pd.DataFrame({'Count':[exoplanet_count,normal_count]},index=['Exoplanet','Non-exoplanet']))
        with right:
            st.subheader('Search the star table')
            dataset_filter=st.selectbox('Dataset',[ 'All datasets', *[name.title() for name in datasets] ],key='star_explorer_dataset')
            search=st.text_input('Search by star number, ID, dataset, or class',key='star_explorer_search')
            filtered=stars if dataset_filter=='All datasets' else stars[stars['Dataset']==dataset_filter]
            if search.strip():
                matches=filtered.astype(str).apply(lambda col:col.str.contains(search.strip(),case=False,regex=False,na=False)).any(axis=1)
                filtered=filtered[matches]
            st.dataframe(filtered,use_container_width=True,height=390,hide_index=True)
            st.caption(f'{len(filtered):,} of {len(stars):,} observations shown.')

        if not filtered.empty:
            selected_index=st.selectbox(
                'Select a star to inspect',
                options=filtered.index.tolist(),
                format_func=lambda i:f"{filtered.loc[i,'Dataset']} · Star #{filtered.loc[i,'Star #']} · {filtered.loc[i,'Classification']}",
                key='star_explorer_selection',
            )
            if st.button('View Light Curve',type='primary',key='star_explorer_view'):
                selected=filtered.loc[selected_index]
                source_name=selected['Dataset'].lower()
                row_index=int(selected['Star #'])-1
                source= datasets[source_name]
                flux,_=prepare(source)
                st.plotly_chart(plot_flux(flux.iloc[row_index],f"Star #{row_index+1} • {source_name.title()} light curve"),use_container_width=True)
                st.write(f"**Classification:** {selected['Classification']}")
                st.caption('The horizontal axis is measurement index; this labeled CSV does not include observation timestamps.')
        else:
            st.info('No observations match this search. Try a different term or dataset.')

elif page=='Light Curves':
    datasets={k:st.session_state[k] for k in ('train','test') if st.session_state[k] is not None}
    if not datasets: st.warning('Upload a dataset in Data Library first.')
    else:
        choice=st.selectbox('Dataset',list(datasets),format_func=str.title)
        data=datasets[choice];x,y=prepare(data)
        idx=st.number_input('Star row (1-based)',min_value=1,max_value=len(data),value=1,step=1)-1
        st.plotly_chart(plot_flux(x.iloc[idx],f'Star #{idx+1} • {choice.title()} light curve'),use_container_width=True)
        if y is not None: st.write('Dataset label:', 'Exoplanet-host class' if y.iloc[idx] else 'Non-exoplanet class')
        st.caption('The horizontal axis is measurement index; this labeled CSV does not include observation timestamps.')

elif page=='Model Training':
    if st.session_state.train is None: st.warning('Upload training data in Data Library first.')
    else:
        train_x,train_y=prepare(st.session_state.train)
        st.write(f'Training set: **{len(train_x):,} stars** • **{train_x.shape[1]:,} measurements each**')
        st.caption('Random Forest uses summary statistics of each light curve. This baseline is not a validated astronomical transit detector.')
        n=st.slider('Number of trees',50,300,120,step=10)
        if st.button('Train Random Forest',type='primary'):
            try:
                if train_y.nunique()!=2: raise ValueError('Both classes must appear in the training dataset.')
                model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=False)),('classifier',RandomForestClassifier(n_estimators=n,class_weight='balanced',random_state=42,n_jobs=-1))])
                with st.spinner('Training model...'): model.fit(extract(train_x),train_y)
                st.session_state.model=model;st.session_state.columns=list(train_x.columns)
                st.success('Training complete.')
            except Exception as e: st.error(f'Training failed: {e}')
        if st.session_state.model is not None and st.session_state.test is not None:
            try:
                tx,ty=prepare(st.session_state.test)
                if list(tx.columns)!=st.session_state.columns: raise ValueError('Testing flux columns do not match training columns.')
                pred=st.session_state.model.predict(extract(tx))
                a,b,c,d=st.columns(4)
                a.metric('Accuracy',f'{accuracy_score(ty,pred):.1%}')
                b.metric('Precision',f'{precision_score(ty,pred,zero_division=0):.1%}')
                c.metric('Recall',f'{recall_score(ty,pred,zero_division=0):.1%}')
                d.metric('F1 score',f'{f1_score(ty,pred,zero_division=0):.1%}')
                cm=confusion_matrix(ty,pred,labels=[0,1]);fig=go.Figure(go.Heatmap(z=cm,x=['Non-host','Host'],y=['Non-host','Host'],text=cm,texttemplate='%{text}',colorscale='Blues'))
                fig.update_layout(title='Confusion matrix',template='plotly_white',paper_bgcolor='#ffffff',height=330,xaxis_title='Predicted',yaxis_title='Actual')
                st.plotly_chart(fig,use_container_width=True)
                st.caption('Because positive examples are rare, prioritize recall, precision and F1 over accuracy alone.')
            except Exception as e: st.error(f'Evaluation unavailable: {e}')

elif page=='Predictions':
    if st.session_state.model is None: st.warning('Train a model in Model Training first.')
    else:
        source=st.selectbox('Prediction dataset',[k for k in ('test','train') if st.session_state[k] is not None])
        try:
            x,y=prepare(st.session_state[source])
            if list(x.columns)!=st.session_state.columns: raise ValueError('Flux columns differ from the training data.')
            features=extract(x);model=st.session_state.model
            pred=model.predict(features);prob=model.predict_proba(features)[:,1]
            results=pd.DataFrame({'star_row':np.arange(1,len(x)+1),'prediction':np.where(pred==1,'Exoplanet-host class','Non-host class'),'host_probability':prob,'actual_label':np.where(y==1,'Exoplanet-host class','Non-host class')})
            st.dataframe(results,use_container_width=True,height=380)
            st.download_button('Download predictions CSV',results.to_csv(index=False).encode(),file_name='kepler_predictions.csv',mime='text/csv')
            idx=st.number_input('Inspect star row',1,len(x),1)-1
            st.plotly_chart(plot_flux(x.iloc[idx],f'Star #{idx+1}'),use_container_width=True)
            st.write(f'**Predicted:** {results.iloc[idx]["prediction"]} • **Model probability:** {prob[idx]:.1%}')
            st.caption('Probability is not calibrated confidence or confirmation of a planet.')
        except Exception as e: st.error(f'Prediction unavailable: {e}')

else:
    st.markdown('### Exoplanet Transit Detection Using Machine Learning')
    st.write('An educational machine learning application for exploring Kepler-derived stellar light curves, training a Random Forest classifier, and analyzing exoplanet-host predictions.')

    st.markdown('### Course Details')
    st.markdown('**Course:** Machine Learning for Data Analytics')
    st.markdown('**Course Code:** 21CSC307P')
    st.markdown('**Project Type:** Final Project')
    st.markdown('**Institution:** SRM Institute of Science and Technology')

    st.markdown('### Project Contributors')
    st.dataframe(pd.DataFrame({
        'Student Name': ['Pranees Yuvaraj', 'Ravish Mugelan S', 'Sridharan D'],
        'Register Number': ['RA2411056010003', 'RA2411056010024', 'RA2411056010010']
    }), use_container_width=True, hide_index=True)

    st.markdown('### Dataset & Methodology')
    st.write('The application supports Kepler-derived labeled time-series CSV files and uses a Random Forest classification baseline.')
    st.markdown('**Dataset:** https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data')
    st.info('This dataset is derived from Kepler observations, not the raw NASA Kepler FITS archive. Model predictions are not astronomical confirmations of exoplanets.')
