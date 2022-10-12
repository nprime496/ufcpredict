
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model,predict_model,blend_models
import shap
import streamlit.components.v1 as components
from sklearn.ensemble import VotingClassifier

@st.cache
def load_data():
    return pd.read_csv("fighters_20_03_2021.csv")

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def preprocess(dataframe):
  data = dataframe.copy()
  data['Men_or_women'] = data.weight_class.str.lower().str.contains('women').astype(int)
  data['R_total_time_fought(mins)']=data['R_total_time_fought(seconds)']/60
  data['B_total_time_fought(mins)']=data['B_total_time_fought(seconds)']/60
  data['R_total_fights'] = data['R_wins']+data['R_draw']+data['R_losses']
  data['B_total_fights'] = data['B_wins']+data['B_draw']+data['B_losses']
  def home_definer(a,b):
    if a=="unknown" or b=="unknown":
      return "dunno"
    if a==b:
      return "yes"
    return "no"
    
  data['R_fighter_home'] = data.apply(lambda x:home_definer(x['R_fighter_country'],x['country_location']),axis=1)
  data['B_fighter_home'] = data.apply(lambda x:home_definer(x['B_fighter_country'],x['country_location']),axis=1)

  data['fighter_taller_but_not_rangier'] = data.apply(lambda x:(x['R_Height_cms']>x['B_Height_cms'] and x['R_Reach_cms']<=x['B_Reach_cms']),axis=1)
  # time
  data['B_avg_time_fought(mins)'] = data['B_total_time_fought(mins)']/(data['B_total_fights']+1)
  data['R_avg_time_fought(mins)'] = data['R_total_time_fought(mins)']/(data['R_total_fights']+1)
  # over fights (win and losses)
  data['R_ratio_win_over_fights_exp']=(data['R_wins']/(data['R_total_fights']+1))*np.exp(data['R_total_fights']/4)
  data['B_ratio_win_over_fights_exp']=(data['B_wins']/(data['B_total_fights']+1))*np.exp(data['B_total_fights']/4)
  data['R_ratio_win']=(data['R_wins']/(data['R_total_fights']+1))
  data['B_ratio_win']=(data['B_wins']/(data['B_total_fights']+1))
  data['R_ratio_losses']=data['R_losses']/(data['R_total_fights']+1)*np.exp(data['R_total_fights']/4)
  data['B_ratio_losses']=data['B_losses']/(data['B_total_fights']+1)*np.exp(data['B_total_fights']/4)
  data['Underdog'] = ((data['R_current_win_streak']>=2) & ~(data['B_current_win_streak']>=2)).astype(int)
  #data['Underdog_lose'] = (data['R_current_lose_streak']<=data['B_current_lose_streak']).astype(int)
  numerical_columns = list(data.select_dtypes(include=['int64','float64']).columns.values)
  print(numerical_columns)
  win_columns = [col[2:] for col in numerical_columns if ('win' in col.lower() or 'lose' in col.lower() )and col.startswith('B_') and 'ratio' not in col]
  
  numerical_columns_fighter = [col[2:] for col in numerical_columns if col.startswith('B_')]
  for col in set(numerical_columns_fighter)-set(win_columns)-{'age'}:
    data[col+'_diff'] = (data['R_'+col]/(data['R_total_fights']+1))-(data['B_'+col]/(data['B_total_fights']+1))
    #data[col+'_ratio'] = (data['R_'+col]*data['R_total_fights'])/(data['B_'+col]*data['B_total_fights']+1)
    numerical_columns.extend([col+'_diff'])#,col+'_ratio'])

  for f in ['R_','B_']:
    for col in win_columns:
      data[f+col+'_over_fights'] = data[f+col]/(data[f+'total_fights']+1)
      numerical_columns.append(f+col+'_over_fights')
  data['Weight_lbs_diff2'] = data['B_Weight_lbs']-data['R_Weight_lbs']

  data['Weight_lbs_diff2_ratio'] = data['Weight_lbs_diff2']/data[['R_Weight_lbs','B_Weight_lbs']].max(axis=1)
  diff = np.log(data['R_age']-17)-np.log(data['B_age']-17) #17 because at 18 years old it will be 0
  data['age_diff2']=diff#*(np.abs(diff)>np.abs(np.log(32/24)))
  data['age_diff_my_ratio']=(data['age_diff2'])/data[['B_age','R_age']].max(axis=1)

  numerical_columns.remove('R_age')
  numerical_columns.remove('B_age')

  return data.drop(columns=['R_age','B_age']),numerical_columns

st.title('UFC FIGHTERS MACHINE LEARNING PREDICTION')
st.subheader("Junior N.")
all_athletes = ''

athlete = 'deiveson-figueiredo'

fighters = []



col1, col2 = st.columns(2)

content =  requests.get(f"https://www.ufc.com/athletes").content
soup = BeautifulSoup(content , features="lxml")
#print(soup)
athletes = soup.find_all(class_='ath-n__name ath-lf-fl')
liste_athletes = [(a.find('a').text.strip(),a.find('a').get('href')) for a in athletes ]
liste_athletes = dict(liste_athletes)

athlete1 = col1.selectbox(
    'Choose Red fighter?',
    tuple(liste_athletes.keys()))
#col1.write('Fighter 1:', athlete1)

athlete2 = col2.selectbox(
    'Choose Blue fighter?',
    tuple(liste_athletes.keys()))
#col2.write('Fighter 2:', athlete2)

selected_ = [(col1,athlete1,'Red'),(col2,athlete2,'Blue')]
for col,athlete,color in selected_:
    #input()
    content =  requests.get(f"https://www.ufc.com{liste_athletes[athlete]}").content
    soup = BeautifulSoup(content,features="lxml")
    #print(soup)
    img = soup.find(class_='hero-profile__image')
    #print(img)
    img_url = img.get('src')

    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    #fighters.append(img)
    name = " ".join([e.capitalize() for e in athlete.split(" ")])
    new_title = f"<p style=\"font-family:sans-serif; color:{color}; font-size: 30px;\">{name}</p>"
    col.markdown(new_title,unsafe_allow_html=True)
    col.image(img, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


fighters_dataset = load_data()

fighters_dataset = fighters_dataset.set_index('fighter')
fighters_dataset.index = [i.lower() for i in fighters_dataset.index]

#rer
st.dataframe(fighters_dataset.loc[[athlete1.lower(),athlete2.lower()]])

fighter1_stats = fighters_dataset.loc[[athlete1.lower()]]
fighter1_stats.columns = [ 'R_'+col for col in fighter1_stats.columns]
fighter2_stats = fighters_dataset.loc[[athlete2.lower()]]
fighter2_stats.columns = [ 'B_'+col for col in fighter2_stats.columns]
st.text(f'Red fighter : {athlete1}, Blue fighter :{athlete2}')

merged_stats = pd.concat([fighter1_stats,fighter2_stats],axis=1)
merged_stats['title_bout'] = False
merged_stats['weight_class'] = 'Heavyweight'
merged_stats['country_location'] = 'usa'

merged_stats = merged_stats.reset_index(drop=True).loc[[0]]

#numerical_columns = list(merged_stats.select_dtypes(include=['int64','float64']).columns.values)


data1,numerical_columns = preprocess(merged_stats)

mylgbm = load_model('mylgbm_normal')
#mylgbm2 = load_model('mylgbm_inverse')

#blender = blend_models([mylgbm2, mylgbm])

#combinedlgbm = VotingClassifier(estimators=[
#        ('normal', mylgbm.named_steps["trained_model"]), ('inverse', mylgbm2.named_steps["trained_model"])], voting='soft')


test_transformed = mylgbm[:-1].transform(data1)
#test_transformed2 = mylgbm2[:-1].transform(data)


explainer = shap.TreeExplainer(mylgbm.named_steps["trained_model"]) #mylgbm.named_steps["trained_model"] not used yet because we don't want finalized model (aka trained on validation)
shap_values = explainer.shap_values(test_transformed)
# Worst predictions on validation set
prediction = predict_model(mylgbm,data1)[['Score','Label']]#+descr_columns+['R_total_fights','B_total_fights']]
#comparison = pd.concat([valid_['Winner'],prediction],axis=1)

st.dataframe(prediction)
print("expected",explainer.expected_value[1])

fight_idx = 0 

shap_values1 = shap_values

#st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][fight_idx,:], test_transformed.loc[fight_idx,:],link='logit'))


fighter1_stats = fighters_dataset.loc[[athlete1.lower()]]
fighter1_stats.columns = [ 'B_'+col for col in fighter1_stats.columns]
fighter2_stats = fighters_dataset.loc[[athlete2.lower()]]
fighter2_stats.columns = [ 'R_'+col for col in fighter2_stats.columns]
st.text(f'Red fighter : {athlete1}, Blue fighter :{athlete2}')

merged_stats = pd.concat([fighter1_stats,fighter2_stats],axis=1)
merged_stats['title_bout'] = False
merged_stats['weight_class'] = 'Heavyweight'
merged_stats['country_location'] = 'usa'

merged_stats = merged_stats.reset_index(drop=True).loc[[0]]

data2,numerical_columns = preprocess(merged_stats)

test_transformed2 = mylgbm[:-1].transform(data2)


shap_values = explainer.shap_values(test_transformed2)
# Worst predictions on validation set
prediction = predict_model(mylgbm,data2)[['Score','Label']]#+descr_columns+['R_total_fights','B_total_fights']]
#comparison = pd.concat([valid_['Winner'],prediction],axis=1)

st.dataframe(prediction)

fight_idx = 0 
print("expected",explainer.expected_value[1])
st_shap(shap.force_plot(explainer.expected_value[1], (shap_values[1][fight_idx,:]+shap_values1[1][fight_idx,:])/2, test_transformed.loc[fight_idx,:],link='logit'))

#shap_values.values=shap_values.values[:,:,1]
#shap_values.base_values=shap_values.base_values[:,1]
# print(type(shap_values[1]))
# print(type(shap_values[1][fight_idx,:]))

# st_shap(shap.waterfall_plot(shap_values[0]))

#st_shap(shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][fight_idx,:],test_transformed.loc[fight_idx,:]))

