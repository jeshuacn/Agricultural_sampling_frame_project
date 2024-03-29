
# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import altair as alt (Graphs look better in quality than matplotlib)
import random
import math

# Connect to Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

sns.set()
pd.set_option('display.max_rows',159)

# Import your data. 
# Add an argument specifying the index column so it doesn't get treated as a numerical variable
df=pd.read_excel('/content/gdrive/MyDrive/Sampling Frame dataset.xlsx',index_col= 'ID')
df.index = df.index.astype('int16')

df.head()

"""## Data Cleaning"""

# Checking variable types and null values
df.info()

"""We can see that all variables have their corresponding data type. Also, we can see that there are missing values in the 'Irrigation Mode' column.

"""

# Showing sum of missing values
df.isna().sum()

"""From the code above, we can see only 92 values are missing."""

# Uniques values in Irrigation when Irrigation mode is NA
df.Irrigation[df.Irrigation_mode.isna()].unique()

# Missing values count grouped by Irrigation
df[['Irrigation', 'Irrigation_mode']][df.Irrigation_mode.isnull()].fillna(1)\
                    .groupby('Irrigation').count().rename({'Irrigation_mode':'Count'},axis = 1)

#missing_values = df[['Irrigation', 'Irrigation_mode']][df.Irrigation_mode.isnull()].fillna(1)
#pd.pivot_table(data=missing_values, index='Irrigation', aggfunc='count')

"""We can clearly see that 99% of missing values are when there is no irrigation. Therefore we will fill those values with 'Rainfed' or 'Pluvial' because those farmers only rely on rain. But what happens when the farmer has irrigation, and there's a missing value? Let's look closer to determine a better way to fill the null value in this case."""

# Showing missing value when 'Irrigation' = yes
df.loc[(df.Irrigation == 'Yes') & (df.Irrigation_mode.isna())].head()

# Showing unique values for Irrigation_mode when Crop == Gren cabbage
df.Irrigation_mode.loc[df.Crop == 'Green cabbage'].unique()

"""We can see that the missing value is related to the crop Green cabbage, so we looked to see if that crop had different irrigation methods and it seems like the only irrigation mode for this crop is 'Localized'. Therefore we are going to fill the missing value with this."""

df[(df.Irrigation == 'No') & (df.Irrigation_mode.isna() != True)].head()

"""We also noticed that some farmers that have an irrigation mode, but are not active or used in this case.

### Filling null values
"""

# Filling the null values adequately
df.loc[df.Irrigation == 'No','Irrigation_mode'] = df.loc[df.Irrigation == 'No','Irrigation_mode'].fillna('Rainfed')

df.Irrigation_mode.fillna('Localized',inplace = True)

"""## Describing numerical variable (Field_area)"""

# What is the average, minimum and maximum area planted ?
df.Field_area.describe()

"""The average planted area is 1.2855 Ha

The minimum area planted is 0.01 Ha

The Maximum area planted is 20 Ha

The 75% of the area planted are below 1.53 HA

### Is the distribution of the area planted equal across the whole population ?

For us to assume that the distribution of area planted is equally distributed across the whole population, we can use two methods, one is using visualizations to show the distribution of a variable, such as a histogram plot, Which gives us the frequency of occurrence per value in the dataset. 

Another method we can use is statistical inference (Hypothesis Testing). It can give a more objective answer to whether a variable deviates significantly from a normal distribution.
"""

# Histogram plot of Field Area
fig, ax  = plt.subplots(figsize = (10,5))
sns.histplot(df.Field_area,ax =ax, color ='#A68160')\
    .set(title = 'Planted Area Distribution (Hectares)', xlabel = 'Planted area')
plt.show()

# The Shapiro Wilk test
from scipy.stats import shapiro

shapiro(df.Field_area)

"""As we can see clearly in the histogram, it is not even close to a normal distribution, which is confirmed by the Shapiro Wilk test. So we can conclude that the distribution is not equal across the whole population.

(If the P-Value of the Shapiro Wilk Test is larger than 0.05, we assume a normal distribution)

### What percentage of farmers hold 80% of the total area planted?

For this task, let's create a relative frequency table to get the percentage accumulation of planted area up to 80% and then divide the number of farmers by the total number of farmers.
"""

# Percentile 80th
np.percentile(df.Field_area,80)

df.Field_area.sort_values().value_counts(sort = False, normalize = True)\
                    .cumsum().to_frame().query('Field_area < 0.81').shape[0]/len(df)*100

# Cumulative relative frequency up to 80% of the area planted 
comulative_relative_frequency = df.Field_area.sort_values()\
                                .value_counts(sort = False, normalize = True)\
                                .cumsum().to_frame().query('index <= 1.93')

# Number of farmers holding 80% of the total area planted
farmers = len(comulative_relative_frequency)

# Percentage of farmers holding 80% of the total area planted round up to two decimal points.
farmer_percent = round((farmers/len(df))*100,2)

print(f'The percentage of farmers holding 80% of the total area planted is {farmer_percent}%, which is equal to {farmers} farmers')

#@title Pareto chart
from matplotlib.ticker import PercentFormatter

# Frequency table
ftd = df.Field_area.sort_values().value_counts(sort = False).to_frame()
# Add cumulative relative frequency 
ftd['cumperc'] = ftd.Field_area.cumsum()/ftd.Field_area.sum()*100

# Define aesthetics for plot
color1 = '#260F01'
color2 = '#334f1e'
barcolor = '#A68160'
linecolor = '#5CA828'
line_size = 4

# Create basic bar plot
fig, ax = plt.subplots(figsize = (20,5))
ax.bar(ftd.index, ftd.Field_area, color=barcolor,width = 0.1,edgecolor = barcolor)
ax.set_ylabel('Count', color=color1, size = 15)
plt.title('Pareto Chart', size = 20)

# Add xtick on 1.93 point
plt.xticks(list(plt.xticks()[0]) + [1.93])

# Add cumulative percentage line to plot
ax2 = ax.twinx()
ax2.plot(ftd.index, ftd.cumperc, color=linecolor, marker="D", ms=line_size)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.set_ylabel('Cumulative %', color=linecolor,size = 15)

# Specify axis colors
ax.tick_params(axis='y', colors=barcolor)
ax2.tick_params(axis='y', colors=color2)

# Add vertical red line
plt.axvline(x=1.93, color ='r')

# Add horizontal grid line
plt.grid(axis='y', linestyle = '--')

# Display Pareto chart
plt.show()

"""## Describing categorical variables"""

#@title ### How many times does each category occur in each variable?
# Looping into each variable in the data frame
for i in df[df.columns.difference(['Field_area'])].columns:

    # Printing column name
    print("Variable: {}".format(i))

    # Printing the count for each value
    print(df[i].value_counts().rename_axis('Unique Values')\
          .to_frame('Count'))
    # Adding blank line
    print('\n')

#@title ### How much percentage does each category take in its variable?

# Function to help print the value as percentage (%)
def aspercent(column,decimals=2):
    assert decimals >= 0
    return (round(column*100,decimals).astype(str) + "%")

for i in df[df.columns.difference(['Field_area'])].columns:
  
    # Printing column name
    print("Variable: {}".format(i))

    # Printing the count for each value
    print(aspercent(df[i].value_counts(normalize=True)\
                    .rename_axis('Unique Values').\
                    to_frame('Percentage'),decimals=1))
    # Adding blank line
    print('\n')

#@title ### Visualizing our findings

# Function to add values to each bar in barplot
def show_values(axs, orient="v", df=[1], space=.01):
    '''
    Add values to bar plots.
    
    Parameters:
    axs (matplotlib.axes.Axes or numpy.ndarray): The axes object(s) to add values to.
    orient (str, optional): The orientation of the bars. Can be "v" (vertical) or "h" (horizontal). Default is "v".
    space (float, optional): The space between the bar and the value label. Default is 0.01.
    '''
    # Helper function to add values to a single axes object
    def _single(ax):
        if orient == "v":

          # Iterate through the patches (individual bars)
            for p in ax.patches:

               # Calculate the x and y position of the value label
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)

                # Format the value as a string and add it to the plot
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
                
        elif orient == "h":
            for p in ax.patches:

                # Calculate the x and y position of the value label
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)

                # Format the value as a string and add it to the plot
                value = '{:.1f}%'.format(p.get_width()/len(df)*100)
                ax.text(_x, _y, value, ha="left")
                
    # If axs is an array, apply the function to each element
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)

    # If axs is a single axes object, apply the function to it
    else:
        _single(axs)


def visualization(df,cols,graph_type,pal = 'terrain_r'):
  '''
    Generate charts from DataFrame
    
    Args:
        df (DataFrame): Pandas DataFrame.
        cols (index): Index of column names.
        graph_type (string): Desired type of chart.
        pal (string): Seaborn Palette, defaul "terrain_r".
    Returns:
        Plot for each column specified. 
  '''    
  
  # Avoid override DataFrame
  df = df.copy()

  # List to store chart coordinates
  ls = []
  n = math.ceil(len(cols)/2)
  

  if n < 2:
    ls.append((0,0))

  else:  
    for x in range(n):
      #for j in range(2):
      ls.append((x,0))
      ls.append((x,1))  

  # Creating subplots
  fig, axs = plt.subplots(nrows=math.ceil(len(cols)/2), ncols=2, figsize=(18,13))

  # validating chart type
  if graph_type == 'bar': 

    if n < 2:
       for i,j in zip(cols,range(len(ls))):

        # Palette
          unique_vals = len(df[i].unique())
          colors = sns.blend_palette([ sns.color_palette(pal)[1],
                                  sns.color_palette(pal)[2],
                                  sns.color_palette(pal)[4]],
                                  unique_vals)

          data = df[i].value_counts().to_frame()

          # Plot
          ax = sns.barplot(x =str(i), y = data.index, data = data,palette = colors,ax = axs[ls[j][0]])\

          # Show values on barplot
          show_values(ax,'h',df)
          #ax.set_title(i.replace("_"," ") + " Frequency")
          ax.set_title('Number of farmers by '+i.replace("_"," "))
          ax.set_xlabel = 'Count'
          ax.set_ylabel = i.replace("_" , " ")

    else:  

      # Iterating every column to create chart 
      for i,j in zip(cols,range(len(ls))):

        # Palette
          unique_vals = len(df[i].unique())
          colors = sns.blend_palette([ sns.color_palette(pal)[1],
                                  sns.color_palette(pal)[2],
                                  sns.color_palette(pal)[4]],
                                  unique_vals)

          data = df[i].value_counts().to_frame()

          # Plot
          ax = sns.barplot(x =str(i), y = data.index, data = data,palette = colors,ax = axs[ls[j][0],ls[j][1]])\

          # Show values on barplot
          show_values(ax,'h',df)
          #ax.set_title(i.replace("_"," ") + " Frequency")
          ax.set_title('Number of farmers by '+i.replace("_"," "))
          ax.set_xlabel = 'Count'
          ax.set_ylabel = i.replace("_" , " ")

  else:
    
    # Palette
    sns.set_palette(sns.color_palette(pal)[1:])

    if n<2:
      # Iterating every column to create chart 
      for i,j in zip(cols,range(len(ls))):
        
        # Plot
        df[i].value_counts().plot(kind=graph_type,ax = axs[ls[j][0]])\
        .set(title = i.replace("_"," ") + " Count", xlabel = "Count" , ylabel = i.replace("_" , " "))
        
    else:
      # Iterating every column to create chart 
      for i,j in zip(cols,range(len(ls))):
        
        # Plot
        df[i].value_counts().plot(kind=graph_type,ax = axs[ls[j][0],ls[j][1]])\
        .set(title = i.replace("_"," ") + " Count", xlabel = "Count" , ylabel = i.replace("_" , " "))

  plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
  
  # If n is not even delete last subplot
  if len(cols)%2 !=0:
    axs.flat[-1].set_visible(False)

  # Display plot
  plt.show()

# Ploting all columns from dataframe except of 'Field_area' and 'Crop_Group'
visualization(df,df[df.columns.difference(['Field_area','Crop_Group'])].columns,'bar')

"""## Digging deeper into the data"""

#@title ### Which crop has the highest Field area ?
df.groupby(['Crop'],as_index = False)['Field_area'].sum().max()

"""The crop with the highest field area is the Zucchini, with 2233.53 Ha.

### Are the number of farmers planting the same crops correlated to the sum of their planted area?
"""

farmers_crop = df.groupby(['Crop'],as_index = False)['Field_area'].sum()
farmers_crop['number_of_farmers'] = df.groupby(['Crop']).size().values

farmers_crop.head()

farmers_crop.corr()

"""A high correlation exists between the number of farmers planting the same crops and the sum of their planted area."""

#@title #### Visualizing our findings
# Setting our palette to start from the 5th value
palette = sns.color_palette("terrain_r", 31)
palette = palette[5:]

# Scatterplot
sns.scatterplot(x = 'number_of_farmers', y ='Field_area', 
                data = farmers_crop.sort_values(by='number_of_farmers',ascending=False),hue='Crop',palette = palette)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

#@title ### Is the Production mode used the same for the same crops?
df.groupby(['Crop','Production_Mode'])['Production_Mode'].count()

"""As shown in the table above, the production mode is different even for the same crop, except for Endive, Jerusalem artichoke, Kiwat tomato, Turnip, and Watermelon."""

# Percentage of crops that has only one production mode
5/len(df)*100

#@title ### Is the Irrigation mode used the same for the same crop?
df.groupby(['Crop','Irrigation_mode'])['Irrigation_mode'].count()

"""The irrigation mode is the same for 42.31% of the crop groups.

### Who uses Greenhouses and who doesn't?
"""

# Crops that use Greenhouse
green = df[df.Greenhouse != 'No']['Crop'].unique()
print(green)

# Crops that don't use Greenhouse
no = df[df.Greenhouse == 'No']['Crop'].unique()
s = set(green)
no_green = [x for x in no if x not in s]
print(no_green)

#@title #### **Visualizing our findings**:
fig, ax = plt.subplots(figsize=(10,5))
pd.crosstab(df.Crop, df.Greenhouse,normalize = 'index').plot(kind='bar',
                                                             ax=ax,title= 'Crop Yields in Greenhouses: A Normalized Comparison',
                                                              width=1)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.show()

#@title ### Which crops rely only on rain?
# Crops that only rely on rain
no = df.Crop[df.Irrigation == 'No'].unique()
s = set(df.Crop[df.Irrigation == 'Yes'].unique())
no_green = [x for x in no if x not in s]
print(no_green)

"""Endive and Jerusalem artichoke are the only crops that only rely on rain.

**Visualizing our findings**:
"""

#@title
fig, ax = plt.subplots(figsize=(10,5))
pd.crosstab(df.Crop, df.Irrigation,normalize = 'index').plot(kind='bar',ax=ax,title= 'Rain-Dependent Crops')

plt.legend(['Rain dependent','Rain independent'],bbox_to_anchor=(1.04, 1), loc="upper left")

"""### Does the Irrigation mode impact the planted area?"""

df.groupby(['Irrigation_mode'])['Field_area'].agg(['mean','median','sum','count'])

"""Farmers have a more extended terrain when the crops rely on rain."""

df.Field_area.sum()

"""The mixed mode "Gravity, Localized" is the irrigation mode that, on average, most impacts the planted area.

# Sampling

Determining an accurate sample size is an intermediate and very detailed field in statistics. However if you would like to specify your own please do. <br>

We are going to use **200 observations** for all samples in this project.

**Simple Random Sampling:**
This takes a totally random sample from the population. We should run this at least 3 times and compare the 3 against each other. We can also shuffle and sort the population as we run this sample. If we get consistent results it's a good sign.

**Systematic Sampling:**
This follows a *step* while sampling and skips rows equally.
$step =  \frac{N}{n}$ where $N$ is the Population size and $n$ the sample size. <br>
This should also be implemented at least a second time by starting from the second row (0,4,8,...) and then (1,5,9,...). We can also shuffle and sort as mentioned above.

**Replicated Sampling:**
This combines both Random and Systematic sampling. Take equal samples of 100 each using each method and combine them. **Note:** It is better **NOT** take unequal samples.

**Stratified Sampling:**
A bit more advanced. With this method we choose a measure of size for the numerical variable. We recommend using the median. Then split the population into two subsets; one containing rows lower than the median and the other the rows higher or equal to the median. 

This is the formula:

$ns_{h} = n \frac{C_{h} X_h^q}{\sum_{h}C_{h} X_h^q}$  with $q=\frac{1}{2}$ and $C_{h}=\frac{S_{h}}{\bar{Y_{h}}}$

Where:
- $h$= ID of the dataset (1 or 2) <br>
- $n$ = Sample size (200) <br>
- $S$ = Standard deviation of the numerical variable in that specific data subset (one for each strata, $S_1$ and $S_2$) <br>
- $\bar{Y}$ = Mean of the numerical variable in that data subset (one for each strata, $Y_1$ and $Y_2$) <br>
- $X$ = Median of your numerical variable in that data subset (you will have two, $X_1$ and $X_2$) <br>

In the end we will have two new "mini sample sizes" $ns_1$ and $ns_2$ that add up to $n$. Extract a random sample from each subset using its respective mini sample size and combine both in one dataframe. Our stratified sample is now ready !

**Probability Proportional to Size Sampling (PPS):**
This one is the most advanced (more than stratified) and cannot be explained in one page. We count on you to refer to this <a href="https://cdn.who.int/media/docs/default-source/hq-tuberculosis/global-task-force-on-tb-impact-measurement/meetings/2008-03/p20_probability_proportional_to_size.pdf?sfvrsn=51372782_3">document</a>  to grasp the concept !
"""

#@title #### Sampling functions

# Simple Random Sampling
def random_sampling(df, sample_size):
    '''
    Returns a random sampled dataframe of length sample_size.
    
    Parameters:
        df: The dataframe to sample from.
        sample_size: Number of rows to include in the sample.
    Returns:
        A sampled dataframe of length sample_size.
  '''   
    
    return df.sample(sample_size)

# Systematic Sampling
def systematic_sampling(df, sample_size):
    '''
    Returns a systematic sampled dataframe of length sample_size. 
    
    Parameters:
        df: The dataframe to sample from.
        sample_size: used to calculate steps.

    Returns:
        A systematically sampled dataframe of length sample_size. 
    '''     

    # Population size
    N = df.shape[0]

    # Set step size / sampling interval
    sampling_interval = N/sample_size

    # Generate a list of indices for the sample
    sample_indices = [int(i * sampling_interval) for i in range(sample_size)]

  
    #df.iloc[::step]

    return df.iloc[sample_indices]

# Replicated Sampling
def replicated_sampling(df, sample_size):
  '''
    Returns a replicated sample of the input dataframe of length sample_size.
    
    Parameters:
        df: The dataframe to sample from.
        sample_size: Number of rows to include in the sample.
    Returns:
        A replicated sample of the input dataframe with sample_size replicates. 
  '''  
  sample_size = sample_size // 2
  # Random sampling
  random = random_sampling(df,sample_size)

  # Systematic sampling
  systematic = systematic_sampling(df,sample_size)

  return pd.concat([random,systematic],axis = 0)


# Probability Proportional to Size Sampling
def pps_sampling(df, variable,sample_size):
    """Perform PPS sampling on a DataFrame.

      Args:
          df (dataFrame): The DataFrame to sample from.
          variable (str): The variable to base the sampling on. Can be either a numerical or categorical variable.
          sample_size (int): The size of the sample to return.

      Returns:
          A DataFrame containing the selected samples.
      """

    if df[variable].dtype == 'float' or df[variable].dtype == 'int':

        # Calculate the population size (total field area)
        population_size = df[variable].sum()

        # Calculate the sampling interval
        sampling_interval = population_size / sample_size

        # Calculate the probability of each crop being selected
        crop_probabilities = df[variable] / population_size
        
        # Calculate the cumulative sum of the probabilities
        #cum_sum = crop_probabilities.cumsum()
    else:

        # Calculate the population size (total number of rows)
        population_size = df[variable].count()

        # Calculate the sampling interval
        sampling_interval = population_size / sample_size

        # Calculate the probability of each crop being selected
        crop_counts = df[variable].value_counts()
        crop_probabilities = crop_counts / population_size

        # Calculate the cumulative sum of the probabilities
        #cum_sum = crop_probabilities.cumsum()

    # Select a random starting point
    start = df[variable].sample(1).iloc[0]

    # Select the samples
    samples = df[df[variable].eq(start)]

    # Select additional samples until the sample size is reached
    while len(samples) < sample_size:
      
        if variable in df.select_dtypes(include=['float', 'int']).columns:
            # Select the next crop based on its probability
            next_crop = random.choices(df.index, weights=crop_probabilities)[0]
            #next_crop = random.choices(df.index, weights=probabilities, cum_weights=cum_sum)[0]

        else:
            # Select the next crop based on its probability
            next_crop = random.choices(list(crop_counts.index), weights=list(crop_probabilities))[0]
            #next_crop = random.choices(list(crop_counts.index), weights=list(crop_probabilities), cum_weights=list(cum_sum))[0]
            

            
        samples = pd.concat([samples, df[df[variable].eq(next_crop)]])

    # Trim the sample to the desired size
    samples = samples.iloc[:sample_size]

    return samples


# Stratified Sampling
def stratified_sampling(df, sample_size):
    '''
    Returns a stratified sample of the input dataframe.
    
    Parameters:
        df: The dataframe to sample from.
        sample_size: The number of rows to include in the sample.
        stratify_by: The column to stratify the sample by.
    
    Returns:
        A stratified sample of the input dataframe, with approximately equal numbers of rows in each stratum.
    '''

    field_area_median = df.Field_area.median()
    lower = df.loc[df.Field_area< field_area_median]
    upper = df.loc[df.Field_area>= field_area_median]

    lower_std = lower.Field_area.std()
    upper_std = upper.Field_area.std()

    lower_mean = lower.Field_area.mean()
    upper_mean = upper.Field_area.mean()

    lower_median = lower.Field_area.median()
    upper_median = upper.Field_area.median()

    C1 = lower_std/lower_mean
    C2 = upper_std/upper_mean

    lower_sample = round(sample_size*(C1*np.sqrt(lower_median))/(C1*np.sqrt(lower_median)+C2*np.sqrt(upper_median)))
    upper_sample = round(sample_size*(C2*np.sqrt(upper_median))/(C1*np.sqrt(lower_median)+C2*np.sqrt(upper_median)))

    lower_df = lower.sample(lower_sample)
    upper_df = upper.sample(upper_sample)

    strata_df = pd.concat([lower_df,upper_df],axis = 0)

    strata_df = strata_df.reset_index().drop(['ID'],axis = 1)

    return   strata_df, lower, upper, lower_sample, upper_sample, lower_df, upper_df

"""## Analyzing Random Samples

**Were the results consistent? Does the Field_area feature still follow the same distribution?**
"""

# Setting random seed
np.random.seed(500)

# Getting 3 random samples
random1 = random_sampling(df, 200)
random2 = random_sampling(df, 200)
random3 = random_sampling(df, 200)

#@title Visualizing our findings
color = sns.color_palette('terrain_r')[1]
fig,ax = plt.subplots(ncols=3,figsize=(18,4))
sns.histplot(random1.Field_area,kde=True, stat="density", linewidth=0, ax =ax[0],color = color)
sns.histplot(random2.Field_area,kde=True, stat="density", linewidth=0, ax =ax[1],color = color)
sns.histplot(random3.Field_area,kde=True, stat="density", linewidth=0, ax =ax[2], color = color)
plt.show()

"""Using visualization to compare the distribution of the random samples, it looks like they are consistent having the same distribution.

### **Statistical test:**
"""

#@title #### Kolmogorov-Smirnov test:
from scipy.stats import ks_2samp


# Perform the Kolmogorov-Smirnov test
statistic, p_value = ks_2samp(random1.Field_area, random2.Field_area)

print(f"KS statistic: {statistic:.3f}")
print(f"p-value: {p_value:.3f}")

# Interpret the result
if p_value < 0.05:
    print("The samples are likely from different distributions (reject H0)")
else:
    print("The samples are likely from the same distribution (fail to reject H0)")

"""**(KS)** test is a nonparametric test used to determine whether two samples come from the same underlying distribution. """

#@title #### Kruskal-Wallis test:
from scipy.stats import kruskal

# Perform the Kruskal-Wallis test
statistic, p_value = kruskal(random1.Field_area, random2.Field_area, random3.Field_area)

print(f"Kruskal-Wallis statistic: {statistic:.3f}")
print(f"p-value: {p_value:.3f}")

# Interpret the result
if p_value < 0.05:
    print("The samples are likely from different distributions (reject H0)")
else:
    print("The samples are likely from the same distribution (fail to reject H0)")

"""The Kruskal-Wallis test can be used to compare the medians of three or more independent samples."""

#@title Were any crops nonexistent in the Random samples?
samples = [random1.Crop, random2.Crop, random3.Crop]

labels = ['Sample 1', 'Sample 2', 'Sample 3']


fig, ax = plt.subplots()

for i, sample in enumerate(samples):
    ax.bar(i, sample.nunique(), label=labels[i])
    
    # Find the elements in the population that are not in the sample
    in_sample = df.Crop.isin(sample)
    population_not_in_sample = df[~in_sample]
    
    print(f'Values not in {labels[i]}:')
    print(f'{population_not_in_sample.Crop.unique()}\n')

ax.set_xticks(range(len(samples)))
ax.set_xticklabels(labels)

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.show()

"""As we can see, some crops were left out of the samples.

----------------------------------------------------------------------------------------

## Comparing all sampling methods

**Were the results consistent? Does the Field_area feature still follow the same distribution in all samplings?**
"""

systematic = systematic_sampling(df,200)
replicated = replicated_sampling(df, 100)
pps = pps_sampling(df,'Field_area',200)
stratified, lower,upper,lower_sample, upper_sample, lower_df, upper_df = stratified_sampling(df,200)

#@title Visualizing distributions:

color = sns.color_palette('terrain_r')[1]
fig,ax = plt.subplots(ncols=6,figsize=(25,5))

g = sns.histplot(df.Field_area,kde=True, stat="density", linewidth=0, ax =ax[0],color = color)
g.axes.title= plt.text(0.5, 1.0, 'Population', ha='center', va='bottom', transform=g.axes.transAxes)

g = sns.histplot(random2.Field_area,kde=True, stat="density", linewidth=0, ax =ax[1],color = color)
g.axes.title= plt.text(0.5, 1.0, 'Random Sample', ha='center', va='bottom', transform=g.axes.transAxes)

g = sns.histplot(systematic.Field_area,kde=True, stat="density", linewidth=0, ax =ax[2],color = color)
g.axes.title= plt.text(0.5, 1.0, 'Systematic Sample', ha='center', va='bottom', transform=g.axes.transAxes)

g = sns.histplot(replicated.Field_area,kde=True, stat="density", linewidth=0, ax =ax[3],color = color)
g.axes.title= plt.text(0.5, 1.0, 'Replicated Sample', ha='center', va='bottom', transform=g.axes.transAxes)

g = sns.histplot(pps.Field_area,kde=True, stat="density", linewidth=0, ax =ax[4],color = color)
g.axes.title= plt.text(0.5, 1.0, 'PPS Sample', ha='center', va='bottom', transform=g.axes.transAxes)

g = sns.histplot(stratified.Field_area,kde=True, stat="density", linewidth=0, ax =ax[5],color = color)
g.axes.title= plt.text(0.5, 1.0, 'Stratified Sample', ha='center', va='bottom', transform=g.axes.transAxes)

# Give more space between the subplots
plt.subplots_adjust( wspace=0.3)

# Show plots
plt.show()

#@title #### Kruskal-Wallis test:
statistic, p_value = kruskal(random1.Field_area, systematic.Field_area, replicated.Field_area, pps.Field_area,stratified.Field_area)

print(f"Kruskal-Wallis statistic: {statistic:.3f}")
print(f"p-value: {p_value:.3f}")

# Interpret the result
if p_value < 0.05:
    print("The samples are likely from different distributions (reject H0)")
else:
    print("The samples are likely from the same distribution (fail to reject H0)")

"""As we can see from the graphs and from the Kruskal-Wallis test, the distribution differs from each other."""

#@title #### Kolmogorov-Smirnov test

# Perform the Kolmogorov-Smirnov test
statistic, p_value = ks_2samp(systematic.Field_area, df.Field_area)

print(f"KS statistic: {statistic:.3f}")
print(f"p-value: {p_value:.3f}")

# Interpret the result
if p_value < 0.05:
    print("The samples are likely from different distributions (reject H0)")
else:
    print("The samples are likely from the same distribution (fail to reject H0)")

#@title Were any crops nonexistent in the samples?

samples = [random2.Crop,systematic.Crop,replicated.Crop,pps.Crop,stratified.Crop]
labels = ['Random Sample','Systematic Sample','Replicated Sample','PPS Sample','Stratified Sample']

fig, ax = plt.subplots(figsize = (10,5))

for i, sample in enumerate(samples):
    ax.bar(i, sample.nunique(), label=labels[i])
    
    # Find the elements in the population that are not in the sample
    in_sample = df.Crop.isin(sample)
    population_not_in_sample = df[~in_sample]
    
    print(f'Values not in {labels[i]}:')
    print(f'{population_not_in_sample.Crop.unique()}\n')

ax.set_xticks(range(len(samples)))
ax.set_xticklabels(labels)

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# Rotate the x-axis labels
plt.xticks(rotation=45)

plt.show()

"""As we can see, some samples were less representative of the population. Some samples are letting out a more significant number of crops.

### Percentage of Each Crop per Sample
"""

# Random sampling Crop value count
crop_random = random1['Crop'].value_counts().to_frame()

# Systematic sampling Crop value count
crop_systematic = systematic['Crop'].value_counts().to_frame()

# Replicated sampling Crop value count
crop_replicated = replicated['Crop'].value_counts().to_frame()

# PPS sampling Crop value count
crop_pps = pps['Crop'].value_counts().to_frame()

# Stratified sampling Crop value count
crop_stratified = stratified['Crop'].value_counts().to_frame()

# Population Crop value count
crop_population =  df['Crop'].value_counts().to_frame()

#@title #### Plotting Percentage of Each Crop per Sampling method


# defining color palette
unique_vals = len(df['Crop'].unique())
colors = sns.blend_palette([ sns.color_palette('terrain_r')[1],
                        sns.color_palette('terrain_r')[2],
                        sns.color_palette('terrain_r')[4]],
                        unique_vals)

# Plotting percentage for each crop per sampling comparison
fig,ax = plt.subplots(ncols=3,nrows=2,figsize=(25,13))

g = sns.barplot(x= 'Crop', y = crop_population.index,data = crop_population, ax = ax[0][0], palette = colors)
g.axes.title= plt.text(0.5, 1.0, 'Population', ha='center', va='bottom', transform=g.axes.transAxes)
show_values(g,'h',df)
g.axes.set_xlabel('Value Count')
g.axes.set_ylabel('Crop')

g = sns.barplot(x= 'Crop', y = crop_random.index, data = crop_random, ax =ax[0][1], palette = colors)
g.axes.title= plt.text(0.5, 1.0, 'Random Sample', ha='center', va='bottom', transform=g.axes.transAxes)
show_values(g,'h',random1)
g.axes.set_xlabel('Value Count')
g.axes.set_ylabel('Crop')

g = sns.barplot(x= 'Crop', y = crop_systematic.index, data = crop_systematic, ax = ax[0][2],palette = colors)
g.axes.title= plt.text(0.5, 1.0, 'Systematic Sample', ha='center', va='bottom', transform=g.axes.transAxes)
show_values(g,'h',systematic)
g.axes.set_xlabel('Value Count')
g.axes.set_ylabel('Crop')

g = sns.barplot(x= 'Crop', y = crop_replicated.index, data = crop_replicated, ax = ax[1][0],palette = colors)
g.axes.title= plt.text(0.5, 1.0, 'Replicated Sample', ha='center', va='bottom', transform=g.axes.transAxes)
show_values(g,'h',replicated)
g.axes.set_xlabel('Value Count')
g.axes.set_ylabel('Crop')

g = sns.barplot(x= 'Crop', y = crop_pps.index, data = crop_pps, ax = ax[1][1],palette = colors)
g.axes.title= plt.text(0.5, 1.0, 'PPS Sample', ha='center', va='bottom', transform=g.axes.transAxes)
show_values(g,'h',pps)
g.axes.set_xlabel('Value Count')
g.axes.set_ylabel('Crop')

g = sns.barplot(x= 'Crop', y = crop_stratified.index,data = crop_stratified, ax = ax[1][2],palette = colors)
g.axes.title= plt.text(0.5, 1.0, 'Stratified Sample', ha='center', va='bottom', transform=g.axes.transAxes)
show_values(g,'h',stratified)
g.axes.set_xlabel('Value Count')
g.axes.set_ylabel('Crop')

# Give more space between the subplots
plt.subplots_adjust(hspace=0.2, wspace=0.45)

# Show plots
plt.show()

"""Just as we could see, the proportions of crops vary per sample. Being a Systematic sample, the most accurate so far.

## Comparing Samples Statistics
"""

#@title ### Comparing Field_area statistics
pop = df.Field_area.describe()
field_random = random1.Field_area.describe()
field_systematic = systematic.Field_area.describe()
field_replicated = replicated.Field_area.describe()
field_pps = pps.Field_area.describe()
field_stratified = stratified.Field_area.describe()

comparison = pd.DataFrame({'Population':pop,
                           'Random':field_random,'Random/POP':field_random/pop,
                           'Systematic Sample':field_systematic,'Systematic/POP':field_systematic/pop,
                           'Replicated':field_replicated,'Replicated/POP':field_replicated/pop,
                           'PPS':field_pps,'PPS/POP':field_pps/pop,
                           'Stratified':field_stratified,'Stratified/POP':field_stratified/pop})


comparison.style.set_properties(**{'background-color': 'cornsilk'}, subset = ['Random/POP','Systematic/POP','PPS/POP','Replicated/POP','Stratified/POP'])\
                                .set_caption('Field_area Comparison')

#@title #### Comparing the Standard Deviation for each Sampling Method over the Population

def color_one(val):
  if val ==1:
    color = 'brown'
  else:
    color = 'black'
  return 'color: %s' % color

STD = np.array([np.std(df['Field_area']),np.std(random1['Field_area']),np.std(systematic['Field_area']),
            np.std(replicated['Field_area']),np.std(pps['Field_area']),np.std(stratified['Field_area'])])

std0 = STD/np.std(df.Field_area)
std1 = STD/np.std(random1.Field_area)
std2 = STD/np.std(systematic.Field_area)
std3 = STD/np.std(replicated.Field_area)
std4 = STD/np.std(pps.Field_area)
std5 = STD/np.std(stratified.Field_area)

idx = ['Population','Random Sample','Systematic Sample','Replicated Sample','PPS Sample','Stratified Sample']
std_frame = pd.DataFrame({'Population':std0,'Random Sample':std1,'Systematic Sample':std2,'Replicated Sample':std3,'PPS Sample':std4,'Stratified Sample':std5},index = idx)


std_frame = std_frame.style.applymap(color_one)\
            .set_caption('Field Area Standard Deviation Comparison by Sampling Method over the Population')

std_frame

#@title #### Comparing the Mean for each Sampling Method over the Population

mean0 = np.array([np.mean(df['Field_area']),np.mean(random1['Field_area']),np.mean(systematic['Field_area']),
            np.mean(replicated['Field_area']),np.mean(pps['Field_area']),np.mean(stratified['Field_area'])])

mean1 = mean0/np.mean(df.Field_area)
mean2 = mean0/np.mean(random1.Field_area)
mean3 = mean0/np.mean(systematic.Field_area)
mean4 = mean0/np.mean(replicated.Field_area)
mean5 = mean0/np.mean(pps.Field_area)
mean6 = mean0/np.mean(stratified.Field_area)

mean_frame = pd.DataFrame({'Population':mean1,'Random Sample':mean2,'Systematic Sample':mean3,'Replicated Sample':mean4,'PPS Sample':mean5,'Stratified Sample':mean6},index = idx)

mean_frame = mean_frame.style.applymap(color_one)\
            .set_caption('Field Area Mean Comparison by Sampling Method over the Population')

mean_frame

#@title #### Comparing the Median for each Sampling Method over the Population

median0 = np.array([np.median(df['Field_area']),np.median(random1['Field_area']),np.median(systematic['Field_area']),
              np.median(replicated['Field_area']),np.median(pps['Field_area']),np.median(stratified['Field_area'])])

median1 = median0/np.median(df.Field_area)
median2 = median0/np.median(random1.Field_area)
median3 = median0/np.median(systematic.Field_area)
median4 = median0/np.median(replicated.Field_area)
median5 = median0/np.median(pps.Field_area)
median6 = median0/np.median(stratified.Field_area)

median_frame = pd.DataFrame({'Population':median1,'Random Sample':median2,'Systematic Sample':median3,'Replicated Sample':median4,'PPS Sample':median5,'Stratified Sample':median6},index = idx)


median_frame = median_frame.style.applymap(color_one)\
            .set_caption('Field Area Mean Comparison by Sampling Method over the Population')

median_frame

"""From the tables above, we can confirm that the Replicated sample has the closest coefficient of variation to the population, indicating that it is the method that represents the population better for this data.

# Inference

Inference as a definition is straightforward. Create a population from each sample method. For clarity we'll provide the following example :

Since we took a random sample of $200$ of the population, If we have for eg $100$ farmers planting potatoes in the sample, it would probably be $100*\frac{606}{200} = 303$ for the population, where 606 is the population size and 200 the sample size.

- Note: Random, Systematic and Replicated sampling use the same Inference method ( multiply by a probability of $\frac{N}{n}$ as mentioned above). This means that all 606 farmers have the same weight in the population. For Stratified and PPS sampling we **CANNOT** use the same method, we will use at least 2 different probabilities to form a population of the same size as the original.
"""

#@title #### Comparing the Crop Inference from the Samples

P = len(df)/len(random1)

random_Estimated_Population_Crop = random1.Crop.value_counts() * P

systematic_Estimated_Population_Crop = systematic.Crop.value_counts() * P
 
replicated_Estimated_Population_Crop = replicated.Crop.value_counts() * P

# Inference for Stratified sampling
proportion_stratum1 = len(lower)/lower_sample
proportion_stratum2 = len(upper)/upper_sample

s1 = lower_df.Crop.value_counts()*proportion_stratum1 
s2 = upper_df.Crop.value_counts()*proportion_stratum2

s = pd.concat([s1,s2],axis = 1).fillna(0)
stratified_Estimated_Population_Crop = s.iloc[:,0]+s.iloc[:,1]

# Inference for PPS sampling

sampling_proportion = df['Field_area'].sum()/len(pps)

pps_Estimated_Population_Crop = pps.Crop.value_counts() * sampling_proportion

# Get all unique crops from the different value_counts dataframes
#all_crops = set(df.Crop.value_counts().index).union(random_Estimated_Population_Crop.index, systematic_Estimated_Population_Crop.index, replicated_Estimated_Population_Crop.index, pps_Estimated_Population_Crop.index, stratified_Estimated_Population_Crop.index)
all_crops = set(df.Crop.value_counts().index)
# Create a new DataFrame with all the unique crops and fill it with the counts
df_all_crops = pd.DataFrame(index=all_crops)
df_all_crops['Population'] = df.Crop.value_counts()
df_all_crops['Random'] = random_Estimated_Population_Crop
df_all_crops['Systematic'] = systematic_Estimated_Population_Crop
df_all_crops['Replicated'] = replicated_Estimated_Population_Crop
df_all_crops['PPS'] = pps_Estimated_Population_Crop
df_all_crops['Estratified'] = stratified_Estimated_Population_Crop

# Filling null values
df_all_crops.fillna(0,inplace = True)

fig,ax = plt.subplots(figsize = (15,8))
df_all_crops.sort_values(by='Population',ascending = False).plot(kind='bar', color = ['green', 'blue', 'red', 'pink', 'orange', 'grey'], width = 0.8,ax = ax)
plt.title('Population Inference Comparison for Crop by Sample Method',fontsize =15)
plt.xlabel('Crop')
plt.ylabel('Count')
plt.show()

#@title #### Comparing the Irrigation Mode Inference from the Samples

random_Estimated_Population_Irrigation_Mode = random1.Irrigation_mode.value_counts() * P

systematic_Estimated_Population_Irrigation_Mode = systematic.Irrigation_mode.value_counts() * P
 
replicated_Estimated_Population_Irrigation_Mode = replicated.Irrigation_mode.value_counts() * P

# Inference for Stratified sampling
s1 = lower_df.Irrigation_mode.value_counts()*proportion_stratum1 
s2 = upper_df.Irrigation_mode.value_counts()*proportion_stratum2

s = pd.concat([s1,s2],axis = 1).fillna(0)
stratified_Estimated_Population_Irrigation_Mode = s.iloc[:,0]+s.iloc[:,1]

# Inference for PPS sampling
pps_Estimated_Population_Irrigation_Mode = pps.Irrigation_mode.value_counts() * sampling_proportion

# Function to add values to each bar in barplot
def show_values1(axs, orient="v", df=[1], space=.01):
    '''
    Add values to bar plots.
    
    Parameters:
    axs (matplotlib.axes.Axes or numpy.ndarray): The axes object(s) to add values to.
    orient (str, optional): The orientation of the bars. Can be "v" (vertical) or "h" (horizontal). Default is "v".
    space (float, optional): The space between the bar and the value label. Default is 0.01.
    '''
    # Helper function to add values to a single axes object
    def _single(ax):
        if orient == "v":

          # Iterate through the patches (individual bars)
            for p in ax.patches:

               # Calculate the x and y position of the value label
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)

                # Format the value as a string and add it to the plot
                value = '{}'.format(round(p.get_height()))
                ax.text(_x, _y, value, ha="center") 
                                
    # If axs is an array, apply the function to each element
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)

    # If axs is a single axes object, apply the function to it
    else:
        _single(axs)

# Get all unique crops from the different value_counts dataframes
all_modes = set(df.Irrigation_mode.value_counts().index)

# Create a new DataFrame with all the unique crops and fill it with the counts
df_all_modes = pd.DataFrame(index=all_modes)
df_all_modes['Population'] = df.Irrigation_mode.value_counts()
df_all_modes['Random'] = random_Estimated_Population_Irrigation_Mode
df_all_modes['Systematic'] = systematic_Estimated_Population_Irrigation_Mode
df_all_modes['Replicated'] = replicated_Estimated_Population_Irrigation_Mode
df_all_modes['PPS'] = pps_Estimated_Population_Irrigation_Mode
df_all_modes['Estratified'] = stratified_Estimated_Population_Irrigation_Mode

# Filling null values
df_all_modes.fillna(0,inplace = True)

fig,ax = plt.subplots(figsize = (18,8))
show_values1(df_all_modes.sort_values(by='Population',ascending = False).plot(kind='bar', color = ['green', 'blue', 'red', 'pink', 'orange', 'grey'], width = 0.8,ax = ax))
plt.title('Irrigation Mode - Population Inference Comparison by Sample Method',fontsize =15)
plt.xlabel('Irrigation Mode')
plt.ylabel('Count')
plt.show()

#@title #### Comparing the Irrigation Inference from the Samples
random_Estimated_Population_Irrigation = random1.Irrigation.value_counts() * P

systematic_Estimated_Population_Irrigation = systematic.Irrigation.value_counts() * P
 
replicated_Estimated_Population_Irrigation = replicated.Irrigation.value_counts() * P

# Inference for Stratified sampling
s1 = lower_df.Irrigation.value_counts()*proportion_stratum1 
s2 = upper_df.Irrigation.value_counts()*proportion_stratum2

s = pd.concat([s1,s2],axis = 1).fillna(0)
stratified_Estimated_Population_Irrigation = s.iloc[:,0]+s.iloc[:,1]

# Inference for PPS sampling
pps_Estimated_Population_Irrigation = pps.Irrigation.value_counts() * sampling_proportion


# Get all unique crops from the different value_counts dataframes
all_irrigation = set(df.Irrigation.value_counts().index)

# Create a new DataFrame with all the unique crops and fill it with the counts
df_all_irrigation = pd.DataFrame(index=all_irrigation)
df_all_irrigation['Population'] = df.Irrigation.value_counts()
df_all_irrigation['Random'] = random_Estimated_Population_Irrigation
df_all_irrigation['Systematic'] = systematic_Estimated_Population_Irrigation
df_all_irrigation['Replicated'] = replicated_Estimated_Population_Irrigation
df_all_irrigation['PPS'] = pps_Estimated_Population_Irrigation
df_all_irrigation['Estratified'] = stratified_Estimated_Population_Irrigation

# Filling null values
df_all_irrigation.fillna(0,inplace = True)

fig,ax = plt.subplots(figsize = (15,8))
show_values1(df_all_irrigation.sort_values(by='Population',ascending = False).plot(kind='bar', color = ['green', 'blue', 'red', 'pink', 'orange', 'grey'], width = 0.8,ax = ax))
plt.title('Irrigation - Population Inference Comparison by Sample Method',fontsize =15)
plt.xlabel('Irrigation')
plt.ylabel('Count')
plt.show()

#@title #### Comparing the Green House Inference from the Samples

random_Estimated_Population_Greenhouse = random1.Greenhouse.value_counts() * P

systematic_Estimated_Population_Greenhouse = systematic.Greenhouse.value_counts() * P
 
replicated_Estimated_Population_Greenhouse = replicated.Greenhouse.value_counts() * P

# Inference for Stratified sampling
s1 = lower_df.Greenhouse.value_counts()*proportion_stratum1 
s2 = upper_df.Greenhouse.value_counts()*proportion_stratum2

s = pd.concat([s1,s2],axis = 1).fillna(0)
stratified_Estimated_Population_Greenhouse = s.iloc[:,0]+s.iloc[:,1]

# Inference for PPS sampling
pps_Estimated_Population_Greenhouse = pps.Greenhouse.value_counts() * sampling_proportion


# Get all unique crops from the different value_counts dataframes
all_Greenhouse = set(df.Greenhouse.value_counts().index)

# Create a new DataFrame with all the unique crops and fill it with the counts
df_all_Greenhouse = pd.DataFrame(index=all_Greenhouse)
df_all_Greenhouse['Population'] = df.Greenhouse.value_counts()
df_all_Greenhouse['Random'] = random_Estimated_Population_Greenhouse
df_all_Greenhouse['Systematic'] = systematic_Estimated_Population_Greenhouse
df_all_Greenhouse['Replicated'] = replicated_Estimated_Population_Greenhouse
df_all_Greenhouse['PPS'] = pps_Estimated_Population_Greenhouse
df_all_Greenhouse['Estratified'] = stratified_Estimated_Population_Greenhouse

# Filling null values
df_all_Greenhouse.fillna(0,inplace = True)

fig,ax = plt.subplots(figsize = (15,8))
show_values1(df_all_Greenhouse.sort_values(by='Population',ascending = False).plot(kind='bar', color = ['green', 'blue', 'red', 'pink', 'orange', 'grey'], width = 0.8,ax = ax))
plt.title('Greenhouse - Population Inference Comparison by Sample Method',fontsize =15)
plt.xlabel('Greenhouse')
plt.ylabel('Count')
plt.show()

"""# Dimensionality Reduction & Factor Analysis

We will use **Multiple Correspondence Analysis (MCA)** to plot all 6 variables on a 2D plot. It allows us to group individuals with similar profiles and check associations between variable categories. For this analysis, we will use R because they have some libraries that are easier to use than python.
"""

df2 = df.copy() # to avoid override the original dataframe

# Creating bins for the numerical variable
df2["Field_area"] = pd.cut(df["Field_area"], [0, 2, 5, 10.5, float("inf")], labels=["Small", "Medium", "Large", "Very large"])

# New categorical variable distribution
df2.Field_area.value_counts()

# Commented out IPython magic to ensure Python compatibility.
# # Downgrading Python-R bridge
# !pip install -q rpy2==3.5.1
# 
# %%R
# 
# # installing package
# install.packages("factoextra")
# install.packages("FactoMineR")
# install.packages("knitr")
# install.packages("fastcluster")
# install.packages("dendextend")
# 
# # importing user dependencies
# library(dplyr)
# library(FactoMineR)
# library(factoextra)
# library(ca)
# library(ggplot2)
# library(knitr)
# library(fastcluster)
# library(dendextend)

# Commented out IPython magic to ensure Python compatibility.
# Loading Python-R extension to execute R code in python environment
# %load_ext rpy2.ipython

# Pass df2 to R
# %R -i df2

# Commented out IPython magic to ensure Python compatibility.
# # Visualize eigenvalues/variances
# %%R
# #MCA_ca=mjca(dfr2)
# MCA_factominer=MCA(df2,ncp=5,graph=F)
# fviz_screeplot(MCA_factominer, addlabels = TRUE, ylim = c(0, 9))
# 
#

"""### MCA - Biplot"""

# Commented out IPython magic to ensure Python compatibility.
# # Biplot of individuals and variable categories
# %%R
# 
# fviz_mca_biplot(MCA_factominer,repel=TRUE,ggtheme=theme_minimal())

"""From the graph above, we can notice which variables are more related to others. For example, we can see that the Strawberry crop is strongly related to farmers that have large amounts of terrain and uses the Irrigation mode "Gravity, Localized."
"""

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variables
# %%R
# 
# fviz_mca_var(MCA_factominer, choice='mca.cor',repel=TRUE,ggtheme=theme_minimal())

"""The farmers are more likely to use to plant the same or similar crops if they have similar terrain lengths and use the same irrigation mode."""

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variable Categories
# %%R
# 
# fviz_mca_var(MCA_factominer,repel=TRUE,ggtheme=theme_minimal())

"""We can see from the graph above that the crops: Carrot, Endive and Jerusalem Artichoke have in common a large field area.

## Clustering
"""

#@title ### Dendongram

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

df3 = df2.copy()
df3.set_index(df3.Crop, inplace = True)

# Data set
m2 = pd.get_dummies(df2[df2.columns.difference(['Crop_Group','Irrigation_mode','Greenhouse'])])
# Calculate the distance between each sample
Z = linkage(m2, 'ward')
 
# Plot with Custom leaves
fig,ax  = plt.subplots(figsize = (40,10))
dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df3.index, truncate_mode = 'level', p=7)

# Show the graph
plt.show()

"""We can see that there are 3 groups inside our data."""

# Commented out IPython magic to ensure Python compatibility.
# #@title ### Kmeans
# %%R
# 
# coord_var=MCA_factominer$var$coord
# df_kmeans <- scale(coord_var)
# res.hk <-hkmeans(df_kmeans, 3)
# fviz_cluster(res.hk, frame.type = "norm", frame.level = 0.68) + theme_bw()

"""Using K means we can see that indeed there are 3 different groups in our data and we can visualize which variables correspond to each group.

### Random sample MCA - Clustering
"""

random11 = random1.copy() # to avoid override the original dataframe

# Creating bins for the numerical variable
random11["Field_area"] = pd.cut(random11["Field_area"], [0, 2, 5, 10.5, float("inf")], labels=["Small", "Medium", "Large", "Very large"])

# Commented out IPython magic to ensure Python compatibility.
# Pass dataframe to R
# %R -i random11

# Commented out IPython magic to ensure Python compatibility.
# # Visualize eigenvalues/variances
# %%R
# #MCA_ca=mjca(dfr2)
# MCA_random=MCA(random11,ncp=5,graph=F)
# fviz_mca_biplot(MCA_random,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variables
# %%R
# 
# fviz_mca_var(MCA_random, choice='mca.cor',repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variable Categories
# %%R
# 
# fviz_mca_var(MCA_random,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title ### Kmeans
# %%R
# 
# coord_var=MCA_random$var$coord
# df_kmeans <- scale(coord_var)
# res.hk <-hkmeans(df_kmeans, 3)
# fviz_cluster(res.hk, frame.type = "norm", frame.level = 0.68) + theme_bw()

"""## Stratified Sample MCA - Clustering"""

stratified1 = stratified.copy() # to avoid override the original dataframe

# Creating bins for the numerical variable
stratified1["Field_area"] = pd.cut(stratified1["Field_area"], [0, 2, 5, 10.5, float("inf")], labels=["Small", "Medium", "Large", "Very large"])

# Commented out IPython magic to ensure Python compatibility.
# Pass dataframe to R
# %R -i stratified1

# Commented out IPython magic to ensure Python compatibility.
# # Visualize eigenvalues/variances
# %%R
# #MCA_ca=mjca(dfr2)
# MCA_strata=MCA(stratified1,ncp=5,graph=F)
# 
# fviz_mca_biplot(MCA_strata,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variables
# %%R
# 
# fviz_mca_var(MCA_strata, choice='mca.cor',repel=TRUE,ggtheme=theme_minimal())
#

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variable Categories
# %%R
# 
# fviz_mca_var(MCA_strata,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title ### Kmeans
# %%R
# 
# coord_var=MCA_strata$var$coord
# df_kmeans <- scale(coord_var)
# res.hk <-hkmeans(df_kmeans, 3)
# fviz_cluster(res.hk, frame.type = "norm", frame.level = 0.68) + theme_bw()

"""## Replicated Sample MCA - Clustering"""

# Commented out IPython magic to ensure Python compatibility.
replicated1 = replicated.copy() # to avoid override the original dataframe

# Creating bins for the numerical variable
replicated1["Field_area"] = pd.cut(replicated1["Field_area"], [0, 2, 5, 10.5, float("inf")], labels=["Small", "Medium", "Large", "Very large"])

# Pass dataframe to R
# %R -i replicated1

# Commented out IPython magic to ensure Python compatibility.
# # Visualize eigenvalues/variances
# %%R
# #MCA_ca=mjca(dfr2)
# MCA_repli=MCA(replicated1,ncp=5,graph=F)
# 
# fviz_mca_biplot(MCA_repli,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variables
# %%R
# 
# fviz_mca_var(MCA_repli, choice='mca.cor',repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variable Categories
# %%R
# 
# fviz_mca_var(MCA_repli,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title ### Kmeans
# %%R
# 
# coord_var=MCA_repli$var$coord
# df_kmeans <- scale(coord_var)
# res.hk <-hkmeans(df_kmeans, 3)
# fviz_cluster(res.hk, frame.type = "norm", frame.level = 0.68) + theme_bw()

"""## PPS Sample MCA - Clustering"""

# Commented out IPython magic to ensure Python compatibility.
pps1 = pps.copy() # to avoid override the original dataframe

# Creating bins for the numerical variable
pps1["Field_area"] = pd.cut(pps1["Field_area"], [0, 2, 5, 10.5, float("inf")], labels=["Small", "Medium", "Large", "Very large"])

# Pass dataframe to R
# %R -i pps1

# Commented out IPython magic to ensure Python compatibility.
# # Visualize eigenvalues/variances
# %%R
# #MCA_ca=mjca(dfr2)
# MCA_pps=MCA(pps1,ncp=5,graph=F)
# 
# fviz_mca_biplot(MCA_pps,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variables
# %%R
# 
# fviz_mca_var(MCA_pps, choice='mca.cor',repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variable Categories
# %%R
# 
# fviz_mca_var(MCA_pps,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title ### Kmeans
# %%R
# 
# coord_var=MCA_pps$var$coord
# df_kmeans <- scale(coord_var)
# res.hk <-hkmeans(df_kmeans, 3)
# fviz_cluster(res.hk, frame.type = "norm", frame.level = 0.68) + theme_bw()

"""## Systematic Sampling MCA - Clustering"""

# Commented out IPython magic to ensure Python compatibility.
systematic1 = systematic.copy() # to avoid override the original dataframe

# Creating bins for the numerical variable
systematic1["Field_area"] = pd.cut(systematic1["Field_area"], [0, 2, 5, 10.5, float("inf")], labels=["Small", "Medium", "Large", "Very large"])

# Pass dataframe to R
# %R -i systematic1

# Commented out IPython magic to ensure Python compatibility.
# # Visualize eigenvalues/variances
# %%R
# #MCA_ca=mjca(dfr2)
# MCA_systematic=MCA(systematic1,ncp=5,graph=F)
# 
# fviz_mca_biplot(MCA_systematic,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variables
# %%R
# 
# fviz_mca_var(MCA_systematic, choice='mca.cor',repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title #### Variable Categories
# %%R
# 
# fviz_mca_var(MCA_systematic,repel=TRUE,ggtheme=theme_minimal())

# Commented out IPython magic to ensure Python compatibility.
# #@title ### Kmeans
# %%R
# 
# coord_var=MCA_systematic$var$coord
# df_kmeans <- scale(coord_var)
# res.hk <-hkmeans(df_kmeans, 3)
# fviz_cluster(res.hk, frame.type = "norm", frame.level = 0.68) + theme_bw()

"""## Conclusion

After a lengthy analysis of each sampling method, we can conclude that the **Systematic sampling** method is the most convenient for this data, the size of the population and the resources available for conducting the survey.

## References:

[Lonely Octopus](https://www.lonelyoctopus.com/about) 🐙

[PPS Sampling in Python by Aayush Malik](https://chaayushmalik.medium.com/pps-sampling-in-python-b5d5d4a8bdf7)

[How to compare two or more distributions](https://towardsdatascience.com/how-to-compare-two-or-more-distributions-9b06ee4d30bf)


[Stratified random sampling](https://www.investopedia.com/terms/stratified_random_sampling.asp#:~:text=For%20example%2C%20if%20the%20researcher,population%20size\)%20%C3%97%20stratum%20size)


[Inference](https://online.stat.psu.edu/stat200/lesson/1/1.2#paragraph--2297)
"""