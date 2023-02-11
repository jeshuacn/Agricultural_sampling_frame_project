import streamlit as st 
import pandas as pd
import numpy as np
#import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import app_functions as fn # Sampling Functions

#App settings
st.set_page_config(page_title='Agricultural Sampling',
                   page_icon=":sunflower:",
                   layout='wide')

# Scrollbar settings
st.markdown("""
                <html>
                    <head>
                    <style>
                        ::-webkit-scrollbar {
                            width: 10px;
                            }

                            /* Track */
                            ::-webkit-scrollbar-track {
                            background: #f1f1f1;
                            }

                            /* Handle */
                            ::-webkit-scrollbar-thumb {
                            background: #888;
                            }

                            /* Handle on hover */
                            ::-webkit-scrollbar-thumb:hover {
                            background: #555;
                            }
                    </style>
                    </head>
                    <body>
                    </body>
                </html>
            """, unsafe_allow_html=True)

# Title
st.title(""" Agricultural Sampling Frames """)
st.write('App by: Jeshua Céspedes')

# Picture
st.image('app/farm.jpg')
st.write("Credits: [Vecteezy.com](https://www.vecteezy.com/vector-art/2711992-farm-horizontal-landscape-scene-with-red-barn) | [Free License](https://www.vecteezy.com/licensing-agreement)")

# Loading data
@st.cache
def load_data():
  df = pd.read_excel('app/data/Sampling_Frame_dataset.xlsx',index_col = 'ID')
  df.index = df.index.astype('int16')

  # Cleaning data
  df.loc[df.Irrigation == 'No','Irrigation_mode'] = df.loc[df.Irrigation == 'No','Irrigation_mode'].fillna('Rainfed')
  df.Irrigation_mode.fillna('Localized',inplace = True)
  df.Irrigation_mode= df.Irrigation_mode.apply(lambda x: x.replace(","," "))
  return df

df = load_data()
#######################################################################################################
# GOAL
st.markdown("""<span style ="font-size:20px;">**Goal:**</span> evaluate a target population using various sampling techniques to determine the most suitable approach for future surveys.

1. Simple Random Sampling
2. Systematic Sampling
3. Replicated Sampling
4. Probability Proportional to Size Sampling
5. Stratified Sampling""",unsafe_allow_html=True)

########################################################################################################
# DATAFRAME INFO

'### Current census data:' 

row1_1,  row1_2, row1_3,  row1_4, row1_5, row1_6= st.columns(6)
with row1_1:
    n_farmers = len(df)
    str_farmers = "🧑‍🌾 " + str(n_farmers) + " Farmers"
    st.markdown(str_farmers)
with row1_2:
    n_crops = len(np.unique(df.Crop).tolist())
    str_crops = "🌾 " + str(n_crops) + " Crops"
    st.markdown(str_crops)
with row1_3:
    n_greenhouse = len(df.Greenhouse.unique())
    str_greenhouse = "🏡 " + str(n_greenhouse) + " Greenhouses"
    st.markdown(str_greenhouse)
with row1_4:
    n_irrigation = len(df.Irrigation_mode.unique())
    str_irrigation= "💧 " + str(n_irrigation) + " Irrigation Modes"
    st.markdown(str_irrigation)
with row1_5:
    n_prod = len(df.Production_Mode.unique())
    str_prod= "🚜 " + str(n_prod) + " Production Modes"
    st.markdown(str_prod)
with row1_6:
    n_total = df.Field_area.sum()
    str_total= "🌱 " + str(n_total.round(2)) + " Total area planted"
    st.markdown(str_total)

    
########################################################################################################

st.markdown("") # Add blank row
see_data = st.expander('You can click here to see the data and its description 👉')
mk = fn.mk # Data Description markdown
with see_data:
    data_col, description_col = st.columns(2)
 
    with data_col:
        '### Data: '
        
        st.dataframe(data = df,use_container_width=True)
        
    with description_col:
        '### Description:'

        st.markdown(f"<div style='height: 380px; overflow-y: scroll; overflow-x: scroll;'><pre>{mk}</pre></div>", unsafe_allow_html = True)
    
##########################################################################

######################################################################################################
#SIDEBAR options

# Plot width and hight adjust
st.sidebar.markdown("**Here you can resize the plots:** 👇")
width = st.sidebar.slider("Plot width", 1, 25, 8)
height = st.sidebar.slider("Plot height", 1, 25, 5)

# Rerun button in sidebar
st.sidebar.markdown("**Here you can perform a recaclulation of the sampling methods:**")
bt = st.sidebar.button('Recalculate Samples')
if bt:

    st.experimental_memo.clear()
    st.experimental_rerun()

############################################################################################################   
# PLOT FUNCTIONS 
def show_plots(data,col):
    '''
    Pots the population data
    
    Parameters:
        data: The dataframe to plot from.
        col: selected column to plot.
    '''   
    fig, ax = plt.subplots(figsize = (width,height))

    pal = 'terrain_r'
    # Palette
    unique_vals = len(data[str(col)].unique())
    colors = sns.blend_palette([ sns.color_palette(pal)[1],
                            sns.color_palette(pal)[2],
                            sns.color_palette(pal)[4]],
                            unique_vals)

    data = data[str(col)].value_counts().to_frame()
    if col == 'Field_area':
      
        #sns.histplot(df.Field_area,ax =ax, color ='#A68160')\
        #    .set(title = 'Planted Area Distribution (Hectares)', xlabel = 'Planted area')
        g = sns.histplot(df.Field_area,kde=True, stat="density", linewidth=0,color = '#A68160')
        g.axes.title= plt.text(0.5, 1.0, 'Planted Area Distribution (Hectares)', ha='center', va='bottom', transform=g.axes.transAxes)
        g.set_xlabel('Planted Area')

    
    else:
        ax = sns.barplot(x =str(col), y = data.index, data = data,palette = colors)

        # Show values on barplot
        fn.show_values(ax,'h',df)
        
        ax.set_title('Number of farmers by '+str(col).replace("_"," "))
        plt.xlabel('Count')
        plt.ylabel(str(col).replace("_" , " "))
    
    # Show plot
    st.pyplot(fig)

def show_plots_inf(data,col,sampling_method = None):
    '''
    Pots the population data
    
    Parameters:
        data: The dataframe to plot from.
        col: selected column to plot.
        sampling_method: sampling method used to plot
    '''   
    fig, ax = plt.subplots(figsize = (width,height))

    pal = 'terrain_r'
    # Palette
    unique_vals = len(data.index.unique())
    colors = sns.blend_palette([ sns.color_palette(pal)[1],
                            sns.color_palette(pal)[2],
                            sns.color_palette(pal)[4]],
                            unique_vals)

    if col == 'Field_area':
        d = {'Random Sampling':random,'Systematic Sampling':systematic, 'Replicated Sampling':replicated, 'PPS Sampling':pps, 'Stratified Sampling':stratified}
        data = d.get(sampling_method)
        #sns.histplot(data.Field_area,ax =ax, color ='#A68160')\
        #    .set(title = f'{sampling_method} - Planted Area Distribution (Hectares)', xlabel = 'Planted area')
        g = sns.histplot(data.Field_area,kde=True, stat="density", linewidth=0,color = '#A68160')
        g.axes.title= plt.text(0.5, 1.0, f'{sampling_method} - Planted Area Distribution (Hectares)', ha='center', va='bottom', transform=g.axes.transAxes)
        g.set_xlabel('Planted Area')

       
    else:
        sns.barplot(x =str(col), y = data.index, data = data,palette = colors)
        # Show values on barplot
        fn.show_values(ax,'h',df)
        #ax.set_title(i.replace("_"," ") + " Frequency")
        ax.set_title('Number of farmers by '+str(col).replace("_"," ")+" (Inference)")
        plt.xlabel('Count')
        plt.ylabel(str(col).replace("_" , " "))
     
        
    # Show plot
    st.pyplot(fig)
    
##############################################################################################
# POPULATION PLOT

row2_1, row2_2 = st.columns(2)
with row2_1:
    st.markdown('## Population')
    'Select a variable to graph:'

    var = {'Crop':'Crop','Greenhouse':'Greenhouse', 'Irrigation':'Irrigation', 'Irrigation mode':'Irrigation_mode', 'Production mode':'Production_Mode','Field area':'Field_area'}    

    col_selection = st.selectbox ("", var.keys(),key = 'columns_count1',label_visibility = 'collapsed')

    show_plots(df,var.get(col_selection))
###############################################################################################
# GETTING THE SAMPLES AND PLOTING THE INFERENCE

sample_size = 200
random = fn.random_sampling(df,sample_size)
systematic = fn.systematic_sampling(df,sample_size)
replicated = fn.replicated_sampling(df, sample_size)
pps = fn.pps_sampling(df,'Field_area',sample_size)
stratified, lower,upper,lower_sample, upper_sample, lower_df, upper_df = fn.stratified_sampling(df,200)


def population_inference(sample_size,col_selection):
    '''
    This function returns the population sampling inference for all sampling methods (random, systematic, replicated, stratified, pps).
    
    Parameters:
        sample_size (int): The sample size used to calculate the population sampling inference.
        col_selection (str): A column name to be included in the calculation.
    
    Returns:
        dataframe: A dataframe for each sampling method, containing the population sampling inference.
    '''   

    P = len(df)/sample_size

    randoms= random[str(col_selection)].value_counts() * P

    systematics = systematic[str(col_selection)].value_counts() * P
    
    replicateds = replicated[str(col_selection)].value_counts() * P

    # Inference for Stratified sampling
    proportion_stratum1 = len(lower)/lower_sample
    proportion_stratum2 = len(upper)/upper_sample

    s1 = lower_df[str(col_selection)].value_counts()*proportion_stratum1 
    s2 = upper_df[str(col_selection)].value_counts()*proportion_stratum2

    s = pd.concat([s1,s2],axis = 1).fillna(0)
    stratifieds = s.iloc[:,0]+s.iloc[:,1]
    stratifieds.sort_values(ascending = False, inplace = True)
    # Inference for PPS sampling

    sampling_proportion = df['Field_area'].sum()/len(pps)

    ppss = pps[str(col_selection)].value_counts() * sampling_proportion

    return randoms, systematics, replicateds, stratifieds, ppss


with row2_2:
    st.markdown('## Sampling and Inference')
    'Select a sampling method to compare with the population and the selected variable:'

    
    show_me_plots_sampling = st.selectbox ("", ['Random Sampling','Systematic Sampling', 'Replicated Sampling', 'PPS Sampling', 'Stratified Sampling'],key = 'columns_count2',label_visibility = 'collapsed')
    random_inf, systematic_inf, replicated_inf, stratified_inf, pps_inf = population_inference(sample_size,var.get(col_selection))
    
    
    if show_me_plots_sampling == 'Random Sampling':
        
        show_plots_inf(random_inf.to_frame(),var.get(col_selection),show_me_plots_sampling)
        
        
    elif show_me_plots_sampling == 'Systematic Sampling':

        show_plots_inf(systematic_inf.to_frame(),var.get(col_selection),show_me_plots_sampling)


    elif show_me_plots_sampling == 'Replicated Sampling':
        
        show_plots_inf(replicated_inf.to_frame(),var.get(col_selection),show_me_plots_sampling)
       

    elif show_me_plots_sampling == 'PPS Sampling':

        show_plots_inf(pps_inf.to_frame(),var.get(col_selection),show_me_plots_sampling)
         

    elif show_me_plots_sampling == 'Stratified Sampling':

        show_plots_inf(stratified_inf.to_frame(),var.get(col_selection),show_me_plots_sampling)
     
    else:
        pass

######################################################################################################
# CALCULATING MISSING VALUES FROM POPULATION

def not_in(col, sampling,selection):
    '''
    This function returns a list of values that are missing in the sample for the selected column.
    
    Parameters:
        col (str): selected column.
        sampling (dataframe): selected sample dataframe.
        selection (str): selected option.
    
    Returns:
        list: list of values not present in the sample from the population.
    '''   


    samples = [sampling[str(col)]]
    d = {'Random Sampling':'Random Sample','Systematic Sampling':'Systematic Sample', 'Replicated Sampling':'Replicated Sample', 'PPS Sampling':'PPS Sample', 'Stratified Sampling':'Stratified Sample'}
    labels = [d.get(selection)]

    
    for i, sample in enumerate(samples):
    
    # Find the elements in the population that are not in the sample
        in_sample = df[str(col)].isin(sample)
        population_not_in_sample = df[~in_sample]
        
        if population_not_in_sample.empty == False:
            st.write(f'##### **{col.replace("_"," ")}s not in {labels[0]}:**')
            st.write(', '.join(population_not_in_sample[str(col)].unique()))


if col_selection != 'Field area':
    if show_me_plots_sampling == 'Random Sampling':
        
        not_in(var.get(col_selection), random,show_me_plots_sampling)

    elif show_me_plots_sampling == 'Systematic Sampling':

        not_in(var.get(col_selection), systematic,show_me_plots_sampling)

    elif show_me_plots_sampling == 'Replicated Sampling':
        
        not_in(var.get(col_selection), replicated,show_me_plots_sampling)

    elif show_me_plots_sampling == 'PPS Sampling':

        not_in(var.get(col_selection), pps,show_me_plots_sampling)

    elif show_me_plots_sampling == 'Stratified Sampling':
        not_in(var.get(col_selection), stratified,show_me_plots_sampling)
        
    else:
        pass

####################################################################################################
def color_one(val):
    '''
    This function add color brown to values equal to one.
    '''
    if val ==1:
        color = 'brown'
    else:
        color = 'black'
    return 'color: %s' % color

####################################################################################################
# STD
STD = np.array([np.std(df['Field_area']),np.std(random['Field_area']),np.std(systematic['Field_area']),
            np.std(replicated['Field_area']),np.std(pps['Field_area']),np.std(stratified['Field_area'])])

std0 = STD/np.std(df.Field_area)
std1 = STD/np.std(random.Field_area)
std2 = STD/np.std(systematic.Field_area)
std3 = STD/np.std(replicated.Field_area)
std4 = STD/np.std(pps.Field_area)
std5 = STD/np.std(stratified.Field_area)

idx = ['Population','Random Sample','Systematic Sample','Replicated Sample','PPS Sample','Stratified Sample']
std_frame = pd.DataFrame({'Population':std0,'Random Sample':std1,'Systematic Sample':std2,'Replicated Sample':std3,'PPS Sample':std4,'Stratified Sample':std5},index = idx)


std_frame = std_frame.style.applymap(color_one)\
            .set_caption('Field Area Standard Deviation Comparison by Sampling Method over the Population')

# Mean
mean0 = np.array([np.mean(df['Field_area']),np.mean(random['Field_area']),np.mean(systematic['Field_area']),
            np.mean(replicated['Field_area']),np.mean(pps['Field_area']),np.mean(stratified['Field_area'])])

mean1 = mean0/np.mean(df.Field_area)
mean2 = mean0/np.mean(random.Field_area)
mean3 = mean0/np.mean(systematic.Field_area)
mean4 = mean0/np.mean(replicated.Field_area)
mean5 = mean0/np.mean(pps.Field_area)
mean6 = mean0/np.mean(stratified.Field_area)

mean_frame = pd.DataFrame({'Population':mean1,'Random Sample':mean2,'Systematic Sample':mean3,'Replicated Sample':mean4,'PPS Sample':mean5,'Stratified Sample':mean6},index = idx)

mean_frame = mean_frame.style.applymap(color_one)\
            .set_caption('Field Area Mean Comparison by Sampling Method over the Population')

# Median
median0 = np.array([np.median(df['Field_area']),np.median(random['Field_area']),np.median(systematic['Field_area']),
              np.median(replicated['Field_area']),np.median(pps['Field_area']),np.median(stratified['Field_area'])])

median1 = median0/np.median(df.Field_area)
median2 = median0/np.median(random.Field_area)
median3 = median0/np.median(systematic.Field_area)
median4 = median0/np.median(replicated.Field_area)
median5 = median0/np.median(pps.Field_area)
median6 = median0/np.median(stratified.Field_area)

median_frame = pd.DataFrame({'Population':median1,'Random Sample':median2,'Systematic Sample':median3,'Replicated Sample':median4,'PPS Sample':median5,'Stratified Sample':median6},index = idx)


median_frame = median_frame.style.applymap(color_one)\
            .set_caption('Field Area Mean Comparison by Sampling Method over the Population')

####################################################################################################
# COMPARISON

st.write('') # add blank row
st.subheader('Comparing statistics for each Sampling Method over the Population')

#Comparing the Standard Deviation for each Sampling Method over the Population
row3_1,space, row3_2 = st.columns([2.5,0.4,4])
with row3_1:

    st.write('') # add blank row
    st.write('Investigate the coefficient of variation to determine which sample method is the more representative of the population.')
    comparison_table = st.selectbox ("", ['Mean Comparison','Standar Deviation Comparison','Median Comparison'],key = 'mean_std_comparison',label_visibility ='hidden')

with row3_2:

    if comparison_table == 'Standar Deviation Comparison':
        st.write('') # add blank row
        std_frame

    elif comparison_table == 'Mean Comparison':
        st.write('') # add blank row
        mean_frame

    else:
        st.write('') # add blank row
        median_frame
#####################################################################################################
# CONCLUSION

st.subheader('Conclusion:')

st.markdown('After a lengthy analysis of each sampling method, we can conclude that the **Systematic sampling** method is the most convenient for this data, the size of the population and the resources availables for conducting the survey.')



''
''
''
''
''
''
''
''
st.markdown('For a comprehensive understanding, please refer to the full analysis: [Notebook](https://colab.research.google.com/github/jeshuacn/Agricultural_sampling_frame_project/blob/main/SamplingFrame.ipynb)' )
st.markdown('<img  alt="GitHub" width="26px" src="https://user-images.githubusercontent.com/3369400/139448065-39a229ba-4b06-434b-bc67-616e2ed80c8f.png" style="padding-right:10px;" />[Github](https://github.com/jeshuacn/Agricultural_sampling_frame_project/blob/main/README.md) ',unsafe_allow_html=True)
#################################################################################################

st.markdown('Special thanks to Lonely Octopus :octopus:')